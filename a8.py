from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")


test_training_data =  [
    ([.9,.8,.6,.3,.1],[1]),
    ([.8,.8,.4,.6,.4],[1]),
    ([.7,.2,.4,.6,.3],[1]),
    ([.5,.5,.8,.4,.8],[0]),
    ([.3,.1,.6,.8,.8],[0]),
    ([.6,.3,.4,.3,.6],[0])
]
test_data = [
    ([1,1,1,.1,.1]),
    ([.5,.2,.1,.7,.7]),
    ([.8,.3,.3,.3,.8]),
    ([.8,.3,.3,.8,.3]),
    ([.9,.8,.8,.3,.6])
]
test = NeuralNet(5,8,1)
test.train(test_training_data)
print(f"Weights:{test.get_ho_weights()}")
print(test.test_with_expected(test_training_data))
for person in test_data:
    print(round((test.evaluate(person)[0])))


