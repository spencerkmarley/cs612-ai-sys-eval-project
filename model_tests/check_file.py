def check_file(model):
    if str(model)[len(str(model))-3:len(str(model))] == ".pt":
        print("Model is a PyTorch file")
    else:
        print("Please select a PyTorch file")
    return 0