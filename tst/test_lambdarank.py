import unittest
from lambdarank import LambdaRankLoss
import torch


class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 2)
        self.fc3 = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

class TestLambdarank(unittest.TestCase):
    def lambdas_sample_test(self, y, s, lambdas, ndcg_at=30, bce_weight=0.0, remove_batch_dim=False):
        y_true = torch.tensor(y)
        y_pred = torch.tensor(s)
        expected_lambdas = torch.tensor(lambdas)
        if remove_batch_dim:
            shape = len(y_true[0][0]), len(y_true[0])
        else:
            shape = len(y_true[0]), len(y_true)
        loss = LambdaRankLoss(shape[0], shape[1], 1, ndcg_at,
                                   bce_grad_weight=bce_weight, remove_batch_dim=remove_batch_dim)
        y_pred.requires_grad = True  # This makes sure that y_pred is watched for gradients
        loss_val = loss(y_pred, y_true)
        loss_val.backward()  # This computes the gradient of loss_val with respect to y_pred
        lambdas_lambdarank = y_pred.grad  #
        self.assertTrue(torch.allclose(expected_lambdas, lambdas_lambdarank, rtol=1e-03))


    def test_get_lambdas(self):
        self.lambdas_sample_test([[1, 1, 1, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1, 0, 1]],
                                 [[0.394383, 0.79844, 0.197551, 0.76823, 0.55397, 0.628871, 0.513401, 0.916195, 0.717297, 0.606969],
                                  [0.0738184, 0.758257, 0.502675, 0.0370191, 0.985911, 0.0580367, 0.669592, 0.666748, 0.830348, 0.252756]],
                                 [[-0.06920932, -0.30873558, -0.058492843, 0.3047775, -0.088394105, 0.17826326,
                                   0.10690676, -0.35733327, 0.23414889, 0.13135682],
                                  [0.050959602, 0.20272574, 0.08800437, 0.05100822, -0.3893178, 0.05126973, 0.28663573,
                                   -0.33881214, 0.24236104, -0.08703362]], 5, 0.5)
        self.lambdas_sample_test([[1, 1, 1, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1, 0, 1]],
                                 [[0.394383, 0.79844, 0.197551, 0.76823, 0.55397, 0.628871, 0.513401, 0.916195, 0.717297, 0.606969],
                                  [0.0738184, 0.758257, 0.502675, 0.0370191, 0.985911, 0.0580367, 0.669592, 0.666748, 0.830348, 0.252756]],
                                 [[-0.0981529, -0.586437, -0.0719084, 0.54124, -0.140295, 0.291305, 0.151254, -0.686095, 0.401098, 0.197991],
                                  [0.050076, 0.337357, 0.113701, 0.051092, -0.751471, 0.0510903, 0.507125, -0.643697, 0.41508, -0.130353]], 5)

        self.lambdas_sample_test([[0, 0, 1, 1]], [[0.1, 0.3, 1, 0]], [[0.175801, 0.190922, -0.366723, 0]], 1)
        self.lambdas_sample_test([[0, 0, 1, 1]], [[0.1, 0.3, 1, 0]], [[0.175801, 0.190922, -0.366723, 0]], 1)
        self.lambdas_sample_test([[0, 0, 1, 1]], [[0.1, 0.3, 1, 0]], [[0.101338, 0.346829, -0.211393, -0.236774]], 2)
        self.lambdas_sample_test([[0, 0, 1, 1], [0, 0, 1, 1]],
                                [[0.1, 0.3, 1, 0], [0.1, 0.3, 1, 0]],
                                [[0.101338, 0.346829, -0.211393, -0.236774], [0.101338, 0.346829, -0.211393, -0.236774]]
                                 , 2)


        self.lambdas_sample_test([[0, 0, 1, 0]], [[0.5, 0, 0.5, 0]], [[2.59696, 0.0136405, -2.63147, 0.0208627]], 2)
        self.lambdas_sample_test([[0, 0, 1, 0]], [[0.1, 0.3, 1, 0]], [[0.160353, 0.174145, -0.487562, 0.153063]], 1)
        self.lambdas_sample_test([[0, 0, 1, 0], [0, 0, 1, 0]],
                                 [[0.1, 0.3, 1, 0], [0.5, 0, 0.5, 0]],
                                 [[0.160353, 0.174145, -0.487562, 0.153063], [2.59696, 0.0136405, -2.63147, 0.0208627]])
        self.lambdas_sample_test([[0, 0, 1, 0]], [[0.1, 0.3, 1, 0]], [[0.160353, 0.174145, -0.487562, 0.153063]])
        self.lambdas_sample_test([[0, 0, 1, 0]], [[0.5, 0, 0.5, 0]], [[2.59696, 0.0136405, -2.63147, 0.0208627]])

    def test_dcg(self):
            loss = LambdaRankLoss(4, 1, ndcg_at=1)
            res = loss.get_inverse_idcg(torch.tensor([[0.0, 0, 1, 1]]))
            assert res == 1

            loss = LambdaRankLoss(4, 1)
            res = loss.get_inverse_idcg(torch.tensor([[0.0, 0, 0, 1]]))
            assert res == 1

    def test_model_lambdarank(self):
            model = TestModel()
            loss_fn = LambdaRankLoss(2, 2, sigma=1)
            optimizer = torch.optim.Adam(model.parameters())

            X = torch.tensor([[0, 0], [1, 0]], dtype=torch.float32)
            Y = torch.tensor([[1, 0],  [0, 1]], dtype=torch.float32)

            model.train()
            for epoch in range(1000):
                optimizer.zero_grad()
                outputs = model(X)
                loss = loss_fn(outputs, Y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                result = model(X)
                assert(result[0,0] > result[0,1])
                assert(result[1,0] < result[1,1])






if __name__ == '__main__':
    unittest.main()