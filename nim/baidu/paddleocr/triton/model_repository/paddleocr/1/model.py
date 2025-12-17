from __future__ import annotations

import json
from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils  # type: ignore[reportMissingImports]
from paddleocr.inference import format_result, load_image
from paddleocr.settings import ServiceSettings

try:
    from paddleocr import PaddleOCRVL
except ImportError as exc:
    raise RuntimeError(
        "paddleocr is missing in the Triton python backend environment"
    ) from exc


class TritonPythonModel:
    def initialize(self, args: dict | None) -> None:
        _ = args
        self.settings = ServiceSettings()
        self.ocr = PaddleOCRVL(
            use_layout_detection=self.settings.use_layout_detection,
            use_chart_recognition=self.settings.use_chart_recognition,
            format_block_content=self.settings.format_block_content,
            vl_rec_backend=self.settings.vl_rec_backend,
            vl_rec_server_url=self.settings.vl_rec_server_url,
        )

    def _error_response(self, message: str) -> pb_utils.InferenceResponse:
        error = pb_utils.TritonError(message)
        return pb_utils.InferenceResponse(output_tensors=[], error=error)

    def execute(
        self, requests: List[pb_utils.InferenceRequest]
    ) -> List[pb_utils.InferenceResponse]:
        responses: List[pb_utils.InferenceResponse] = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            if input_tensor is None:
                responses.append(self._error_response("INPUT tensor missing"))
                continue

            try:
                payloads = input_tensor.as_numpy().tolist()
            except Exception as exc:
                responses.append(
                    self._error_response(f"Failed to read INPUT tensor: {exc}")
                )
                continue

            outputs: List[bytes] = []
            failed = False
            for raw in payloads:
                try:
                    url = (
                        raw.decode("utf-8")
                        if isinstance(raw, (bytes, bytearray))
                        else str(raw)
                    )
                    image = load_image(url, self.settings.request_timeout_seconds)
                    result = self.ocr.predict(
                        [image],
                        use_layout_detection=self.settings.use_layout_detection,
                        use_chart_recognition=self.settings.use_chart_recognition,
                        format_block_content=self.settings.format_block_content,
                    )[0]
                    formatted = format_result(result)
                    outputs.append(json.dumps(formatted.model_dump()).encode("utf-8"))
                except Exception as exc:
                    responses.append(self._error_response(f"Inference failed: {exc}"))
                    failed = True
                    break

            if failed:
                continue

            out_tensor = pb_utils.Tensor("OUTPUT", np.array(outputs, dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
