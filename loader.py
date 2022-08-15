import architecture

def load_model(name, arg):
    model_name = getattr(architecture, name)
    model = model_name(**arg)
    return model