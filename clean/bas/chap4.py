from fastbook import *
my_list = [*range(1, 40)]
list_comp = [x * 2 for x in my_list if x % 2 != 0]
list_comp

dat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
my_tensor = tensor(dat) * 2

my_tensor[1:, 1:]

# Question 26
# Could probably use the python equivs of assert to check the lengths are the same


def pair_em(a_list, a_string):
    return [(a_list[i], a_string[i]) for i in range(len(a_list))]


a_list = [1, 2, 3, 4]
a_string = "abcd"

pair_em(a_list, a_string)


def rmse(preds, truth)


# Question 33: training loop steps
for x, y in dl:
    pred = model(x)
    loss = loss_func(pred, y)
    loss.backward()  # because loss_func uses pred, and pred took params, and params had required_grads, so it modifies in place
    parameters -= parameters.grad * lr


# p173 outlines code for a basic training loop of an epoch
def train_epoch(model, lr, params):
    for xy, yb in dl:  # does this refer to a global?
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()  # modifying in place


for i in range(20):
    train_epoch(model, lr, params)
