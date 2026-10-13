#!/usr/bin/env python3
"""Convert the published chunk=32 Zipformer ONNX triplet to RKNN."""

import argparse
from pathlib import Path

import onnx
from rknn.api import RKNN


def get_metadata_and_input_sizes(filename: Path):
    model = onnx.load(str(filename), load_external_data=False)
    metadata = ";".join(
        f"{item.key}={item.value}" for item in model.metadata_props
    )
    input_names = [value.name for value in model.graph.input]
    input_sizes = []
    for value in model.graph.input:
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                dims.append(dim.dim_value)
            elif dim.dim_param:
                # These exported graphs use a single symbolic batch dimension N.
                if dim.dim_param != "N":
                    raise ValueError(
                        f"Unsupported symbolic dimension {dim.dim_param!r} "
                        f"in {filename}:{value.name}"
                    )
                dims.append(1)
            else:
                raise ValueError(
                    f"Unspecified input dimension in {filename}:{value.name}"
                )
        input_sizes.append(dims)
    return metadata, input_names, input_sizes


def convert(
    source: Path,
    target: Path,
    platform: str,
    quantization_dataset: Path | None = None,
):
    metadata, input_names, input_sizes = get_metadata_and_input_sizes(source)
    print(f"\nConverting {source}")
    print(f"inputs={list(zip(input_names, input_sizes))}")
    print(f"custom_string={metadata}")

    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(
            target_platform=platform,
            optimization_level=0,
            custom_string=metadata,
        )
        if ret != 0:
            raise RuntimeError(f"RKNN config failed: {ret}")

        ret = rknn.load_onnx(
            model=str(source), inputs=input_names, input_size_list=input_sizes
        )
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: {ret}")

        quantized = quantization_dataset is not None
        if quantized and not quantization_dataset.is_file():
            raise FileNotFoundError(
                f"Quantization dataset does not exist: {quantization_dataset}"
            )
        ret = rknn.build(
            do_quantization=quantized,
            dataset=str(quantization_dataset) if quantized else None,
        )
        if ret != 0:
            raise RuntimeError(f"RKNN build failed: {ret}")

        target.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(target))
        if ret != 0:
            raise RuntimeError(f"export_rknn failed: {ret}")
    finally:
        rknn.release()

    print(f"Wrote {target} ({target.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-platform", default="rk3568")
    parser.add_argument(
        "--quantization-dataset-dir",
        type=Path,
        help=(
            "Optional directory containing encoder.txt, decoder.txt, and "
            "joiner.txt RKNN calibration lists. Omit to export the existing "
            "non-quantized variant."
        ),
    )
    args = parser.parse_args()

    for component in ("encoder", "decoder", "joiner"):
        convert(
            args.input_dir / f"{component}.onnx",
            args.output_dir / f"{component}.rknn",
            args.target_platform,
            (
                args.quantization_dataset_dir / f"{component}.txt"
                if args.quantization_dataset_dir
                else None
            ),
        )


if __name__ == "__main__":
    main()
