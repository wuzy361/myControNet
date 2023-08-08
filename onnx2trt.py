import tensorrt as trt
import os
from pdb import set_trace


def get_shape_table(num_inputs):

    input_shapes = None
    if num_inputs == 4:
       input_shapes = [
                        [1, 4, 32, 48],
                        [1, 3, 256, 384],
                        [1],
                        [1, 77, 768],
                      ]
    elif num_inputs == 16:
        input_shapes =  [
                            [1, 4, 32, 48],
                            [1],
                            [1, 77, 768],
                            [1, 320, 32, 48],
                            [1, 320, 32, 48],
                            [1, 320, 32, 48],
                            [1, 320, 16, 24],
                            [1, 640, 16, 24],
                            [1, 640, 16, 24],
                            [1, 640, 8, 12],
                            [1, 1280, 8, 12],
                            [1, 1280, 8, 12],
                            [1, 1280, 4, 6],
                            [1, 1280, 4, 6],
                            [1, 1280, 4, 6],
                            [1, 1280, 4, 6],
                       ]

    return input_shapes
    

def build_model(onnxFile):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)
    
    if not os.path.exists(onnxFile):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")

    with open(onnxFile, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")
    

    num_inputs = network.num_inputs
    input_shapes = get_shape_table(num_inputs)
    for x in range(num_inputs):
        profile.set_shape(network.get_input(x).name, input_shapes[x], input_shapes[x],input_shapes[x],)

    config.add_optimization_profile(profile)
    engineString = builder.build_serialized_network(network, config)
    
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    
    with open(onnxFile.split(".")[0] + ".plan", "wb") as f:
        f.write(engineString)
        


if __name__ == "__main__":

    onnxFiles =["control_model.onnx", "diffusion_model.onnx"]
    #onnxFiles =["diffusion_model.onnx"]
    for onnxFile in onnxFiles:
        build_model(onnxFile)
        print("build " +  onnxFile + " done!")
