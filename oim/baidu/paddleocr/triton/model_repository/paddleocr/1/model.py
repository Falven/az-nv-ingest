from __future__ import annotations

from typing import List

import triton_python_backend_utils as pb_utils  # type: ignore[reportMissingImports]
from oim_paddleocr.inference import build_grpc_output, denormalize_chw, format_result
from oim_paddleocr.settings import ServiceSettings

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
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input")
            if input_tensor is None:
                responses.append(self._error_response("input tensor missing"))
                continue

            try:
                payloads = input_tensor.as_numpy()
            except Exception as exc:
                responses.append(
                    self._error_response(f"Failed to read input tensor: {exc}")
                )
                continue

            if payloads.ndim != 4 or payloads.shape[1] != 3:
                responses.append(
                    self._error_response(
                        "input tensor must have shape (batch, 3, height, width)"
                    )
                )
                continue

            results = []
            failed = False
            for raw in payloads:
                try:
                    image = denormalize_chw(raw)
                except Exception as exc:
                    responses.append(
                        self._error_response(f"Failed to decode input: {exc}")
                    )
                    failed = True
                    break

                try:
                    result = self.ocr.predict(
                        [image],
                        use_layout_detection=self.settings.use_layout_detection,
                        use_chart_recognition=self.settings.use_chart_recognition,
                        format_block_content=self.settings.format_block_content,
                    )[0]
                    results.append(
                        format_result(
                            result, image_size=(image.shape[1], image.shape[0])
                        )
                    )
                except Exception as exc:
                    responses.append(self._error_response(f"Inference failed: {exc}"))
                    failed = True
                    break

            if failed:
                continue

            grpc_output = build_grpc_output(results)
            out_tensor = pb_utils.Tensor("output", grpc_output)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
