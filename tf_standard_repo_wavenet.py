from wavenet import WaveNetModel

SONG_VECTOR_SIZE = 24
BATCH_SIZE = 1
BLOCK_SIZE = 4000

USE_GPU = True

STANDARD_SAVE_REPO = "../standard_repo_results/"

def wavenet_loss(audio_batch,global_vector_batch):
    net = WaveNetModel(
        batch_size=BATCH_SIZE,
        #use_biases=False,# consider changing
        filter_width=2,
        dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256,
                   1, 2, 4, 8, 16, 32, 64, 128, 256],
        residual_channels=32,
        dilation_channels=32,
        quantization_channels=128,
        skip_channels=256,
        use_biases=True,
        scalar_input=False,
        initial_filter_width=32,
        global_condition_channels=SONG_VECTOR_SIZE,
        global_condition_cardinality=None # treates global_condition_channels as a simple vector, as it should
    )
    loss = net.loss(input_batch=audio_batch,
                    global_vector=global_vector_batch)
    return loss
