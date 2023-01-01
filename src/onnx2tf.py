if __name__ == '__main__':
    import onnx
    from onnx_tf.backend import prepare

    EXPORT_DIR = '../best_models/resnet3-128-old'

    onnx_model = onnx.load(f'{EXPORT_DIR}/conv_block.onnx')
    tf_model = prepare(onnx_model)
    tf_model.export_graph(f'{EXPORT_DIR}/conv-block.tf')
