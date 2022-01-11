class Config(object):
    def __init__(self):
        self.batch_size = 32
        self.max_visual_len = 100
        self.max_audio_len = 150
        self.max_text_len = 25
        self.epoch_num = 15
        self.hidden_dim = 300
        self.att_dim = 300
        self.class_num = 3
        self.dropout = 0.7
        self.learning_rate = 0.001
        self.miss_rate = 0.2

