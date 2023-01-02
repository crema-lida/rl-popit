if __name__ == '__main__':
    import onnx
    from onnx_tf.backend import prepare

    EXPORT_DIR = '../best_models/resnet3-64'

    for module in ['conv_block', 'policy_head']:
        onnx_model = onnx.load(f'{EXPORT_DIR}/{module}.onnx')
        tf_model = prepare(onnx_model)
        tf_model.export_graph(f'{EXPORT_DIR}/{module}.tf')
