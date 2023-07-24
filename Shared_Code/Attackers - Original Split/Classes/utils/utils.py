from copy import deepcopy

import torch


def getTrainableParameters(model) -> list:
    '''
    model: torch module
    '''
    trainableParam = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainableParam.append(name)
    return trainableParam


"""def getFloatSubModules(Delta) -> list:
    param_float = []
    for param in Delta:
        if "FloatTensor" not in Delta[param].type() or "num_batches_tracked" in param:
            continue
        param_float.append(param)
    return param_float"""

def getFloatSubModules(Delta) -> list:
    param_float = []
    for param in Delta:
        if not "FloatTensor" in Delta[param].type():
            continue
        param_float.append(param)
    return param_float


"""def getNetMeta(Delta) -> (dict, dict):
    '''
    get the shape and number of elements in each modules of Delta
    get the module components of type float and otherwise 
    '''
    shapes = dict(((k, v.shape) for (k, v) in Delta.items()))
    sizes = dict(((k, v.numel()) for (k, v) in Delta.items()))
    return shapes, sizes"""
def getNetMeta(Delta) -> (dict, dict):
    shapes = dict(((k, v.shape) for (k, v) in Delta.items()))
    sizes = dict(((k, v.numel()) for (k, v) in Delta.items()))
    return shapes, sizes


"""def vec2net(vec: torch.Tensor, net): 
    '''
    convert a 1 dimension Tensor to state dict
    
    vec : torch vector with shape([d]), d is the number of elements \
            in all module components specified in `param_name`
    net : the state dict to hold the value
    
    return
    None
    '''
    print("Total size of vec:", vec.numel())
    param_float = getFloatSubModules(net)
    shapes, sizes = getNetMeta(net)
    partition = list(sizes[param] for param in param_float)
    print("Sum of partition:", sum(partition))
    print("param_float:", param_float)
    print("sizes:", sizes)
    print("Contains NaN:", torch.isnan(vec).any())
    print("Contains Infinity:", torch.isinf(vec).any())
    flattenComponents = dict(zip(param_float, torch.split(vec, partition)))
    components = dict(((k, v.reshape(shapes[k])) for (k, v) in flattenComponents.items()))
    net.update(components)
    return net"""


def vec2net(vec, net, sizes, param_float):
    partition = [sizes[i] for i in range(len(param_float))]
    start = 0
    for param_tensor in net.parameters():
        end = start + torch.prod(torch.tensor(param_tensor.size()))
        # Get the corresponding tensor from vec using the partition sizes
        tensor_shape = param_tensor.size()
        tensor_size = torch.prod(torch.tensor(tensor_shape)).item()
        tensor_flat = vec[start:end]
        if tensor_size > 0:  # Avoid zero-dimensional tensors
            # Check if the tensor size matches the expected size from the partition
            if tensor_flat.size(0) == partition[0]:
                param_tensor.data = tensor_flat.view(tensor_shape)
                start = end
            else:
                print(f"Error: Tensor size mismatch. Expected size: {partition[0]}, Actual size: {tensor_flat.size(0)}")
                return
        # Remove the first element from the partition list after using it
        partition.pop(0)
    return net


"""def net2vec(net) -> (torch.Tensor):
    '''
    convert state dict to a 1 dimension Tensor
    
    Delta : torch module state dict
    
    return
    vec : torch.Tensor with shape(d), d is the number of Float elements in `Delta`
    '''
    param_float = getFloatSubModules(net)

    components = []
    for param in param_float:
        components.append(net[param])
    vec = torch.cat([component.flatten() for component in components])
    return vec"""

def net2vec(net):
    components = []
    for param_tensor in net.values():
        components.append(param_tensor.flatten())
    
    print("components:", components)

    vec = torch.cat(components)
    return vec


def applyWeight2StateDicts(deltas, weight):
    '''
    for each submodules of deltas, apply the weight to the n state dict
    
    deltas: a list of state dict, len(deltas)==n
    weight: torch.Tensor with shape torch.shape(n,)
    
    return
        Delta: a state dict with its submodules being weighted by `weight`         
    
    '''
    Delta = deepcopy(deltas[0])
    param_float = getFloatSubModules(Delta)

    for param in param_float:
        Delta[param] *= 0
        for i in range(len(deltas)):
            Delta[param] += deltas[i][param] * weight[i].item()

    return Delta


def stackStateDicts(deltas):
    '''
    stacking a list of state_dicts to a state_dict of stacked states, ignoring non float values
    
    deltas: [dict, dict, dict, ...]
        for all dicts, they have the same keys and different values in the form of torch.Tensor with shape s, e.g. s=torch.shape(10,10)
    
    return
        stacked: dict
            it has the same keys as the dict in deltas, the value is a stacked flattened tensor from the corresponding tenors in deltas. 
            e.g. deltas[i]["conv.weight"] has a shape torch.shape(10,10), 
                then stacked["conv.weight"]] has shape torch.shape(10*10,n), and
                stacked["conv.weight"]][:,i] is equal to deltas[i]["conv.weight"].flatten()
    '''
    stacked = deepcopy(deltas[0])
    for param in stacked:
        stacked[param] = None
    for param in stacked:
        param_stack = torch.stack([delta[param] for delta in deltas], -1)
        shaped = param_stack.view(-1, len(deltas))
        stacked[param] = shaped
    return stacked


if __name__ == "__main__":
    from tasks.cifar import Net

    netA = Net().state_dict()
    netB = Net().state_dict()
    for param in netB:
        netB[param] *= 0


    def getNumUnequalModules(netA, netB):
        count = 0
        for param in netA:
            res = torch.all(netA[param] == netB[param])
            if res != True:
                count += 1
        return count


    print("before conversion")
    print("Number of unequal modules:\t", getNumUnequalModules(netA, netB))

    vec = net2vec(netA)
    vec2net(vec, netB)

    param_float = getFloatSubModules(netA)
    for param in netA:
        if param in param_float:
            continue
        netB[param] = netA[param]

    print("After conversion")
    print("Number of unequal modules:\t", getNumUnequalModules(netA, netB))
