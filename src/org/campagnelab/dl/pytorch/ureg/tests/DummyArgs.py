class DummyArgs:
    def __init__(self, ureg=True, ureg_alpha=1,
                 lr=0.001,
                 shave_lr=0.001) -> None:
        super().__init__()
        self.num_shaving = 30
        self.num_validation = 30
        self.num_training = 30
        self.max_examples_per_epoch = 30
        self.ureg = ureg
        self.threshold_activation_size = None
        self.include_layer_indices = "0"
        self.resume = False
        self.model = "test-model"
        self.lr =lr
        self.shave_lr = shave_lr
        self.ureg_learning_rate = 0.01
        self.ureg_reset_every_n_epoch = None
        self.momentum = 0.9
        self.L2 = 0
        self.ureg_num_features = 3
        self.ureg_alpha = ureg_alpha
        self.constant_learning_rates = False
        self.lr_patience = 10
        self.mode = "one_pass"
        self.shaving_epochs = 1