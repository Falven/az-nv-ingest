from __future__ import annotations

import concurrent.futures as futures
import json
import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from nv_ingest_api.internal.enums.common import (
    ContentDescriptionEnum,
    ContentTypeEnum,
    TableFormatEnum,
    TextTypeEnum,
)
from nv_ingest_api.internal.extract.pdf.engines import nemoretriever as base_parser
from nv_ingest_api.internal.primitives.nim import ModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
    YOLOX_PAGE_IMAGE_FORMAT,
    YOLOX_PAGE_IMAGE_PREPROC_HEIGHT,
    YOLOX_PAGE_IMAGE_PREPROC_WIDTH,
)
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import NemoRetrieverParseConfigSchema
from nv_ingest_api.util.image_processing.transforms import crop_image, numpy_to_base64
from nv_ingest_api.util.metadata.aggregators import (
    Base64Image,
    LatexTable,
    construct_image_metadata_from_pdf_image,
    construct_text_metadata,
    extract_pdf_metadata,
)
from nv_ingest_api.util.nim import create_inference_client

logger = logging.getLogger(__name__)


# Re-export constants from the upstream parser for consistency
NEMORETRIEVER_PARSE_RENDER_DPI = base_parser.NEMORETRIEVER_PARSE_RENDER_DPI
NEMORETRIEVER_PARSE_MAX_WIDTH = base_parser.NEMORETRIEVER_PARSE_MAX_WIDTH
NEMORETRIEVER_PARSE_MAX_HEIGHT = base_parser.NEMORETRIEVER_PARSE_MAX_HEIGHT
NEMORETRIEVER_PARSE_MAX_BATCH_SIZE = base_parser.NEMORETRIEVER_PARSE_MAX_BATCH_SIZE

# Typing helpers
BBox = Tuple[float, float, float, float]
BBoxWithDims = Tuple[BBox, Tuple[int, int]]


def _chunk_list(items: Sequence[Any], chunk_size: int) -> List[List[Any]]:
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), max(1, chunk_size))]


def _normalize_bbox(raw_bbox: Sequence[float], dims: Optional[Tuple[int, int]]) -> BBoxWithDims:
    if raw_bbox is None:
        return (-1.0, -1.0, -1.0, -1.0), (NEMORETRIEVER_PARSE_MAX_WIDTH, NEMORETRIEVER_PARSE_MAX_HEIGHT)

    bbox = tuple(float(x) for x in raw_bbox)
    width, height = dims if dims else (NEMORETRIEVER_PARSE_MAX_WIDTH, NEMORETRIEVER_PARSE_MAX_HEIGHT)

    if max(bbox) > 2 and width > 0 and height > 0:
        normalized = (bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height)
    else:
        normalized = bbox

    return normalized, (int(width), int(height))


def _standardize_layout_element(raw: Dict[str, Any]) -> Dict[str, Any]:
    element_type = raw.get("type") or raw.get("category") or raw.get("label") or ""
    text = raw.get("text") or raw.get("markdown") or raw.get("content") or ""
    layout_id = (
        raw.get("layout_id")
        or raw.get("id")
        or raw.get("block_id")
        or raw.get("table_id")
        or raw.get("span_id")
    )

    dims = raw.get("page_dimensions") or raw.get("page_size")
    if isinstance(dims, (list, tuple)) and len(dims) >= 2:
        dims_tuple: Optional[Tuple[int, int]] = (int(dims[0]), int(dims[1]))
    else:
        width = raw.get("page_width") or raw.get("width")
        height = raw.get("page_height") or raw.get("height")
        dims_tuple = (int(width), int(height)) if width and height else None

    bbox_raw = raw.get("bbox") or raw.get("bounding_box") or raw.get("normalized_bbox")
    bbox_norm, dims_tuple = _normalize_bbox(bbox_raw, dims_tuple)

    page_idx = (
        raw.get("page")
        or raw.get("page_index")
        or raw.get("page_idx")
        or raw.get("pageNumber")
        or raw.get("page_number")
        or 0
    )

    return {
        "type": element_type,
        "text": text,
        "bbox": bbox_norm,
        "bbox_max_dimensions": dims_tuple,
        "layout_id": layout_id,
        "page": int(page_idx),
    }


def _group_layout_by_page(layout: Iterable[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for raw in layout:
        element = _standardize_layout_element(raw)
        grouped[element["page"]].append(element)
    return [grouped[idx] for idx in sorted(grouped.keys())]


def _extract_layout_blocks(response: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    Handle multiple possible Nemotron Parse response shapes and return
    a list-of-list structure: outer list is per page/image, inner list
    contains standardized layout elements.
    """
    if not response:
        return []

    if "layout" in response:
        return _group_layout_by_page(response["layout"] or [])

    choices = response.get("choices") or []
    if choices:
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            try:
                arguments = tool_calls[0]["function"]["arguments"]
                parsed = json.loads(arguments)
                layout = parsed.get("layout") if isinstance(parsed, dict) else parsed
                return _group_layout_by_page(layout or [])
            except Exception:
                logger.debug("Failed to parse tool_call arguments for Nemotron Parse response", exc_info=True)

        for content_item in message.get("content", []):
            if content_item.get("type") in {"output_layout", "layout"}:
                return _group_layout_by_page(content_item.get("layout") or [])
            if content_item.get("type") == "text":
                try:
                    parsed_text = json.loads(content_item.get("text", "{}"))
                    layout = parsed_text.get("layout") if isinstance(parsed_text, dict) else parsed_text
                    if layout:
                        return _group_layout_by_page(layout or [])
                except Exception:
                    continue

    return []


class NemotronParseModelInterface(ModelInterface):
    """
    Model interface that understands Nemotron Parse 1.1 layout responses.
    It remains backward compatible with the older NemoRetriever Parse outputs.
    """

    def __init__(self, model_name: str = "nvidia/nemotron-parse-1.1"):
        self.model_name = model_name

    def name(self) -> str:
        return "nemoretriever_parse"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs):
        if protocol != "http":
            raise ValueError("Nemotron Parse shim only supports HTTP protocol")

        images = data.get("images") or [data.get("image")]
        base64_list = [numpy_to_base64(img) for img in images if img is not None]

        formatted_batches: List[Dict[str, Any]] = []
        formatted_batch_data: List[Dict[str, Any]] = []
        for chunk in _chunk_list(base64_list, max_batch_size):
            messages = [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}
                for b64 in chunk
            ]
            payload = {
                "model": self.model_name,
                "messages": messages,
                # Nemotron Parse 1.1 uses layout-rich responses when output_format=layout.
                "output_format": "layout",
            }
            formatted_batches.append(payload)
            formatted_batch_data.append({})

        return formatted_batches, formatted_batch_data

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs):
        if protocol != "http":
            raise ValueError("Nemotron Parse shim only supports HTTP protocol")
        return _extract_layout_blocks(response)

    def process_inference_results(self, output: Any, **kwargs):
        return output


def _create_clients(config: NemoRetrieverParseConfigSchema):
    model_interface = NemotronParseModelInterface(model_name=config.nemoretriever_parse_model_name)
    return create_inference_client(
        config.nemoretriever_parse_endpoints,
        model_interface,
        config.auth_token,
        config.nemoretriever_parse_infer_protocol,
        config.timeout,
    )


def _bbox_to_pixel_coords(bbox: Sequence[float], dims: Tuple[int, int]) -> List[int]:
    width, height = dims
    return [
        math.floor(bbox[0] * width),
        math.floor(bbox[1] * height),
        math.ceil(bbox[2] * width),
        math.ceil(bbox[3] * height),
    ]


def _build_table_metadata(
    table: LatexTable,
    page_idx: int,
    page_count: int,
    source_metadata: Dict[str, Any],
    base_unified_metadata: Dict[str, Any],
    layout_custom: Optional[Dict[str, Any]],
) -> List[Any]:
    content_type, metadata, uid = base_parser._construct_table_metadata(
        table, page_idx, page_count, source_metadata, base_unified_metadata
    )

    table_metadata = metadata.get("table_metadata", {}) or {}
    custom = table_metadata.get("custom_content", {}) or {}
    if layout_custom:
        custom.update(layout_custom)
    table_metadata["custom_content"] = custom
    metadata["table_metadata"] = table_metadata
    return [content_type, metadata, uid]


def annotate_table_continuity(extracted_data: List[List[Any]]) -> None:
    """
    Annotate table metadata with cross-page continuity hints.
    Mutates extracted_data in-place.
    """
    indexed_tables: List[Tuple[int, Optional[str], int]] = []
    for idx, item in enumerate(extracted_data):
        if not item or item[0] != ContentTypeEnum.STRUCTURED:
            continue
        metadata = item[1] or {}
        subtype = metadata.get("content_metadata", {}).get("subtype")
        if subtype != ContentTypeEnum.TABLE:
            continue
        table_md = metadata.get("table_metadata", {}) or {}
        custom = table_md.get("custom_content", {}) or {}
        table_id = custom.get("layout_id")
        page_number = metadata.get("content_metadata", {}).get("page_number", -1)
        indexed_tables.append((idx, table_id, page_number))

    grouped: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for idx, table_id, page in indexed_tables:
        key = table_id if table_id is not None else f"__missing__::{idx}"
        grouped[key].append((idx, page))

    for key, entries in grouped.items():
        pages = sorted({p for _, p in entries})
        span = {"start": pages[0], "end": pages[-1], "pages": pages}
        spanning = len(pages) > 1
        for idx, page in entries:
            metadata = extracted_data[idx][1]
            table_md = metadata.get("table_metadata", {}) or {}
            custom = table_md.get("custom_content", {}) or {}

            if key.startswith("__missing__"):
                custom["continuity"] = {"status": "cannot_join", "reason": "missing_layout_id"}
            else:
                position = "single"
                if spanning:
                    if page == span["start"]:
                        position = "start"
                    elif page == span["end"]:
                        position = "end"
                    else:
                        position = "middle"
                custom["continuity"] = {
                    "status": "spanning" if spanning else "single",
                    "position": position,
                    "page_span": span,
                }

            table_md["custom_content"] = custom
            metadata["table_metadata"] = table_md
            extracted_data[idx][1] = metadata


def nemotron_parse_extractor(
    pdf_stream,
    extract_text: bool,
    extract_images: bool,
    extract_infographics: bool,
    extract_tables: bool,
    extract_charts: bool,
    extractor_config: dict,
    execution_trace_log: Optional[List[Any]] = None,
) -> List[Any]:
    """
    Drop-in replacement for the upstream nemo retriever parser that consumes
    Nemotron Parse 1.1 layout output and preserves layout metadata.
    """
    row_data = extractor_config.get("row_data")
    if row_data is None:
        raise ValueError("Missing 'row_data' in extractor_config.")

    try:
        source_id = row_data["source_id"]
    except KeyError as exc:
        raise KeyError("row_data must contain 'source_id'.") from exc

    text_depth_str = extractor_config.get("text_depth", "page")
    try:
        text_depth = TextTypeEnum[text_depth_str.upper()]
    except KeyError as exc:
        valid_options = [e.name.lower() for e in TextTypeEnum]
        raise ValueError(f"Invalid text_depth value: {text_depth_str}. Expected one of: {valid_options}") from exc

    extract_tables_method = extractor_config.get("extract_tables_method", "yolox")
    identify_nearby_objects = extractor_config.get("identify_nearby_objects", True)
    table_output_format_str = extractor_config.get("table_output_format", "pseudo_markdown")
    try:
        table_output_format = TableFormatEnum[table_output_format_str.upper()]
    except KeyError as exc:
        valid_options = [e.name.lower() for e in TableFormatEnum]
        raise ValueError(
            f"Invalid table_output_format value: {table_output_format_str}. Expected one of: {valid_options}"
        ) from exc

    nemoretriever_parse_config_raw = extractor_config.get("nemoretriever_parse_config", {})
    nemoretriever_parse_config = (
        NemoRetrieverParseConfigSchema(**nemoretriever_parse_config_raw)
        if isinstance(nemoretriever_parse_config_raw, dict)
        else nemoretriever_parse_config_raw
    )

    metadata_col = extractor_config.get("metadata_column", "metadata")
    if hasattr(row_data, "index") and metadata_col in row_data.index:
        base_unified_metadata = row_data[metadata_col]
    else:
        base_unified_metadata = row_data.get(metadata_col, {})

    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    source_location = base_source_metadata.get("source_location", "")
    collection_id = base_source_metadata.get("collection_id", "")
    partition_id = base_source_metadata.get("partition_id", -1)
    access_level = base_source_metadata.get("access_level")

    extracted_data: List[Any] = []
    doc = base_parser.pdfium.PdfDocument(pdf_stream)
    pdf_metadata = extract_pdf_metadata(doc, source_id)
    page_count = pdf_metadata.page_count

    source_metadata = {
        "source_name": pdf_metadata.filename,
        "source_id": source_id,
        "source_location": source_location,
        "source_type": pdf_metadata.source_type,
        "collection_id": collection_id,
        "date_created": pdf_metadata.date_created,
        "last_modified": pdf_metadata.last_modified,
        "summary": "",
        "partition_id": partition_id,
        "access_level": access_level,
    }

    accumulated_text: List[str] = []
    accumulated_tables: List[Dict[str, Any]] = []
    accumulated_images: List[Base64Image] = []

    pages_for_ocr: List[Tuple[int, np.ndarray]] = []
    pages_for_tables: List[Tuple[int, np.ndarray, Tuple[int, int]]] = []
    pending_futures: List[Any] = []

    nemotron_client = None
    if extract_text:
        nemotron_client = _create_clients(nemoretriever_parse_config)

    max_workers = nemoretriever_parse_config.workers_per_progress_engine
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for page_idx in range(page_count):
            page = doc.get_page(page_idx)
            page_image, padding_offset = base_parser._convert_pdfium_page_to_numpy_for_parser(page)
            pages_for_ocr.append((page_idx, page_image))

            page_image_for_tables, padding_offset_for_tables = base_parser._convert_pdfium_page_to_numpy_for_yolox(page)
            pages_for_tables.append((page_idx, page_image_for_tables, padding_offset_for_tables))
            page.close()

            if extract_text and len(pages_for_ocr) >= NEMORETRIEVER_PARSE_MAX_BATCH_SIZE:
                future_parser = executor.submit(
                    lambda *args, **kwargs: ("parser", base_parser._extract_text_and_bounding_boxes(*args, **kwargs)),
                    pages_for_ocr[:],
                    nemotron_client,
                    execution_trace_log=execution_trace_log,
                )
                pending_futures.append(future_parser)
                pages_for_ocr.clear()

            if (
                extract_tables_method == "yolox"
                and (extract_tables or extract_charts or extract_infographics)
                and len(pages_for_tables) >= base_parser.YOLOX_MAX_BATCH_SIZE
            ):
                future_yolox = executor.submit(
                    lambda *args, **kwargs: ("yolox", base_parser._extract_page_elements(*args, **kwargs)),
                    pages_for_tables[:],
                    page_count,
                    source_metadata,
                    base_unified_metadata,
                    extract_tables,
                    extract_charts,
                    extract_infographics,
                    table_output_format,
                    nemoretriever_parse_config.yolox_endpoints,
                    nemoretriever_parse_config.yolox_infer_protocol,
                    nemoretriever_parse_config.auth_token,
                    execution_trace_log=execution_trace_log,
                )
                pending_futures.append(future_yolox)
                pages_for_tables.clear()

        if extract_text and pages_for_ocr:
            future_parser = executor.submit(
                lambda *args, **kwargs: ("parser", base_parser._extract_text_and_bounding_boxes(*args, **kwargs)),
                pages_for_ocr[:],
                nemotron_client,
                execution_trace_log=execution_trace_log,
            )
            pending_futures.append(future_parser)
            pages_for_ocr.clear()

        if (
            extract_tables_method == "yolox"
            and (extract_tables or extract_charts or extract_infographics)
            and pages_for_tables
        ):
            future_yolox = executor.submit(
                lambda *args, **kwargs: ("yolox", base_parser._extract_page_elements(*args, **kwargs)),
                pages_for_tables[:],
                page_count,
                source_metadata,
                base_unified_metadata,
                extract_tables,
                extract_charts,
                extract_infographics,
                table_output_format,
                nemoretriever_parse_config.yolox_endpoints,
                nemoretriever_parse_config.yolox_infer_protocol,
                nemoretriever_parse_config.auth_token,
                execution_trace_log=execution_trace_log,
            )
            pending_futures.append(future_yolox)
            pages_for_tables.clear()

        parser_results: List[Tuple[int, List[Dict[str, Any]]]] = []
        for fut in futures.as_completed(pending_futures):
            model_name, extracted_items = fut.result()
            if model_name == "yolox" and (extract_tables or extract_charts or extract_infographics):
                extracted_data.extend(extracted_items)
            elif model_name == "parser":
                parser_results.extend(extracted_items)

    for page_idx, parser_output in parser_results:
        page = None
        page_image = None
        page_text: List[str] = []

        page_nearby_blocks = {
            "text": {"content": [], "bbox": [], "type": []},
            "images": {"content": [], "bbox": [], "type": []},
            "structured": {"content": [], "bbox": [], "type": []},
        }

        for bbox_dict in parser_output:
            cls = bbox_dict.get("type")
            bbox = bbox_dict.get("bbox")
            txt = bbox_dict.get("text", "")
            layout_id = bbox_dict.get("layout_id")
            bbox_max_dims = bbox_dict.get("bbox_max_dimensions") or (
                NEMORETRIEVER_PARSE_MAX_WIDTH,
                NEMORETRIEVER_PARSE_MAX_HEIGHT,
            )

            if not cls or bbox is None:
                continue

            transformed_bbox = _bbox_to_pixel_coords(bbox, bbox_max_dims)

            if cls not in base_parser.nemoretriever_parse_utils.ACCEPTED_CLASSES:
                continue

            if identify_nearby_objects:
                base_parser._insert_page_nearby_blocks(page_nearby_blocks, cls, txt, transformed_bbox)

            if extract_text:
                page_text.append(txt)

            if extract_tables_method == "nemoretriever_parse" and extract_tables and cls == "Table":
                table = LatexTable(
                    latex=txt,
                    bbox=transformed_bbox,
                    max_width=bbox_max_dims[0],
                    max_height=bbox_max_dims[1],
                )
                accumulated_tables.append({"table": table, "layout_id": layout_id})

            if extract_images and cls == "Picture":
                if page is None:
                    page = doc.get_page(page_idx)
                if page_image is None:
                    page_image, _ = base_parser._convert_pdfium_page_to_numpy_for_parser(page)

                img_numpy = crop_image(page_image, transformed_bbox)
                if img_numpy is not None:
                    base64_img = numpy_to_base64(img_numpy, format=YOLOX_PAGE_IMAGE_FORMAT)
                    image = Base64Image(
                        image=base64_img,
                        bbox=transformed_bbox,
                        width=img_numpy.shape[1],
                        height=img_numpy.shape[0],
                        max_width=bbox_max_dims[0],
                        max_height=bbox_max_dims[1],
                    )
                    accumulated_images.append(image)

        if not "".join(page_text).strip():
            if page is None:
                page = doc.get_page(page_idx)
            page_text = [page.get_textpage().get_text_bounded()]

        accumulated_text.extend(page_text)

        if extract_tables:
            for table_info in accumulated_tables:
                custom = {"layout_id": table_info.get("layout_id")}
                extracted_data.append(
                    _build_table_metadata(
                        table_info["table"],
                        page_idx,
                        page_count,
                        source_metadata,
                        base_unified_metadata,
                        layout_custom=custom,
                    )
                )
            accumulated_tables = []

        if extract_images:
            for image in accumulated_images:
                extracted_data.append(
                    construct_image_metadata_from_pdf_image(
                        image,
                        page_idx,
                        page_count,
                        source_metadata,
                        base_unified_metadata,
                    )
                )
            accumulated_images = []

        if extract_text and text_depth == TextTypeEnum.PAGE:
            extracted_data.append(
                construct_text_metadata(
                    accumulated_text,
                    pdf_metadata.keywords,
                    page_idx,
                    -1,
                    -1,
                    -1,
                    page_count,
                    text_depth,
                    source_metadata,
                    base_unified_metadata,
                    delimiter="\n\n",
                    bbox_max_dimensions=(NEMORETRIEVER_PARSE_MAX_WIDTH, NEMORETRIEVER_PARSE_MAX_HEIGHT),
                    nearby_objects=page_nearby_blocks,
                )
            )
            accumulated_text = []

    if extract_text and text_depth == TextTypeEnum.DOCUMENT:
        text_extraction = construct_text_metadata(
            accumulated_text,
            pdf_metadata.keywords,
            -1,
            -1,
            -1,
            -1,
            page_count,
            text_depth,
            source_metadata,
            base_unified_metadata,
            delimiter="\n\n",
        )
        if text_extraction:
            extracted_data.append(text_extraction)

    if nemotron_client:
        nemotron_client.close()
    doc.close()

    annotate_table_continuity(extracted_data)
    return extracted_data


def register_nemotron_parse_extractor() -> None:
    """
    Register the Nemotron Parse extractor and model interface so that the existing
    'nemoretriever_parse' method routes through the shim without changing callers.
    """
    try:
        from nv_ingest_api.internal.extract.pdf.engines import pdf_helpers
        from nv_ingest_api.internal.primitives.nim import model_interface as model_iface_pkg

        pdf_helpers.EXTRACTOR_LOOKUP["nemoretriever_parse"] = nemotron_parse_extractor
        pdf_helpers.EXTRACTOR_LOOKUP["nemotron_parse"] = nemotron_parse_extractor

        # Swap the model interface used by the inference client so batching remains untouched.
        model_iface_pkg.nemoretriever_parse.NemoRetrieverParseModelInterface = NemotronParseModelInterface
        logger.info("Nemotron Parse shim registered for PDF extraction.")
    except Exception:
        logger.exception("Failed to register Nemotron Parse shim; falling back to default extractor.")
