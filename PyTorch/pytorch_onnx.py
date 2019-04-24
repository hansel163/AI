import torch
import torchvision


def export_resnet18_onnx(model_file):
    model = torchvision.models.resnet18(pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
    output_names = ["output1"]
    # Invoke export
    torch.onnx.export(
        model, dummy_input, model_file, verbose=True,
        input_names=input_names, output_names=output_names)


def export_alexnet_onnx(model_file):
    # Standard ImageNet input - 3 channels, 224x224,
    # values don't matter as we care about network structure.
    # But they can also be real inputs.
    dummy_input = torch.randn(1, 3, 224, 224)
    # Obtain your model, it can be also constructed in your script explicitly
    model = torchvision.models.alexnet(pretrained=True)
    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
    output_names = ["output1"]
    # Invoke export
    torch.onnx.export(
        model, dummy_input, model_file, verbose=True,
        input_names=input_names, output_names=output_names)


def onnx_check(model_file):
    import onnx
    # Load the ONNX model
    model = onnx.load(model_file)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))


def alexnet():
    export_alexnet_onnx("alexnet.onnx")
    onnx_check("alexnet.onnx")    


def resnet18():
    export_resnet18_onnx("resnet18.onnx")
    onnx_check("resnet18.onnx")    


def main():
    resnet18()

if __name__ == "__main__":
    main()
