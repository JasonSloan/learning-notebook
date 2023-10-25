import logging
import tensorrt as trt

LOGGER = logging.getLogger("ESRGAN")
LOGGER.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
LOGGER.addHandler(handler)

onnx_path = "/root/rrdb.onnx"
trt_path = "/root/rrdb.engine"
logger = trt.Logger(trt.Logger.INFO)
logger.min_severity = trt.Logger.Severity.VERBOSE
builder = trt.Builder(logger)
config = builder.create_builder_config()
config.max_workspace_size = 4 * 1 << 30

flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(flag)
parser = trt.OnnxParser(network, logger)

# Parse the ONNX model
with open(onnx_path, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            LOGGER.error(parser.get_error(error))

# Assuming your network has only one output layer
last_layer = network.get_layer(network.num_layers - 1)
if last_layer is not None:
    last_layer_output = last_layer.get_output(0)
    network.mark_output(last_layer_output)
    LOGGER.info(f'TensorRT building FP{16 if builder.platform_has_fast_fp16 else 32} engine as {trt_path}')

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    with builder.build_engine(network, config) as engine, open(trt_path, 'wb') as t:
        t.write(engine.serialize())
else:
    LOGGER.error("Last layer not found in the network.")


def t1():
    import logging
    import tensorrt as trt

    LOGGER = logging.getLogger("ESRGAN")
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    LOGGER.addHandler(handler)

    trt_path = "/root/rrdb.engine"
    logger = trt.Logger(trt.Logger.INFO)
    logger.min_severity = trt.Logger.Severity.VERBOSE
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 4 * 1 << 30

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    last_layer = network.get_layer(network.num_layers - 1)
    if last_layer is not None:
        last_layer_output = last_layer.get_output(0)
        network.mark_output(last_layer_output)
    profile = builder.create_optimization_profile()
    profile.set_shape('images', (1, 3, 64, 64), (1, 3, 480, 700), (1, 3, 720, 960))
    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open("/root/rrdb.engine", 'wb') as t:
        t.write(engine.serialize())
