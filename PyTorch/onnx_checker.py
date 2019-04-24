import sys
import onnx


def onnx_check(model_file):
    import onnx
    # Load the ONNX model
    model = onnx.load(model_file)
    # Check that the IR is well formed
    print("Checking on ", model_file, " ...")
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    print("Display model graph:")
    print(onnx.helper.printable_graph(model.graph))


def main():
    onnx_check(sys.argv[1])


if __name__ == "__main__":
    main()
