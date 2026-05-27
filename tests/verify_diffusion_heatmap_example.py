from pyside_app.execution_engine import ExecutionEngine
from pyside_app.notebook_examples import get_desktop_notebook_examples
from pyside_app.notebook_plot_panel import (
    build_notebook_heatmap_figure,
    extract_notebook_array_variables_with_3d,
)


def main() -> None:
    examples = get_desktop_notebook_examples()
    example = next(item for item in examples if item.id == "diffusion_heatmap_2d")
    source = next(cell["source"] for cell in example.cells if cell["type"] == "code")

    print(f"[verify][diffusion-example] input: title={example.title!r}")
    engine = ExecutionEngine()
    result = engine.execute(source)
    if result.error:
        raise RuntimeError(result.error)

    namespace = engine.get_namespace()
    arrays_1d, arrays_2d, arrays_3d = extract_notebook_array_variables_with_3d(namespace)
    figure = build_notebook_heatmap_figure(
        arrays_1d,
        arrays_2d,
        arrays_3d,
        "u_history",
        "time_history",
        "x",
        "y",
        True,
        "",
        "",
        "",
    )

    print(f"[verify][diffusion-example] output: u_shape={namespace['u'].shape}")
    print(f"[verify][diffusion-example] output: u_history_shape={namespace['u_history'].shape}")
    print(f"[verify][diffusion-example] output: time_history_shape={namespace['time_history'].shape}")
    print(f"[verify][diffusion-example] output: heatmap_frames={len(figure.frames)}")

    assert namespace["u"].shape == (80, 100)
    assert namespace["u_history"].shape == (100, 80, 100)
    assert namespace["time_history"].shape == (100,)
    assert len(figure.frames) == 100
    assert figure.data[0].zauto is True
    assert figure.frames[1].data[0].zauto is True


if __name__ == "__main__":
    main()
