from gorilla.nn.models.domain_adaptation import DANN

def build_model(cfg):
    if cfg.method == "DANN":
        model = DANN(cfg)

        return model
