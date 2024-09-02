import torch

a = torch.tensor([[0,2,3],[2.5,6,1],[8,3,4]])

i = torch.linalg.inv(a)
print("Torch's inverse")
print(i)

def chebyshev_inverse(a,iterations=10):
    """Computes an approximation of the inverse of a matrix
    alpha = \frac{1}{(||A||_{1})*||A||_{\inf}}
    \begin{cases}
       N(0) = alpha.a^T
       N(t+1) = N(t)(3.In -a\cdotN(t)(3.In - a.N(t)))
    \end{cases}

    -> Notes:
    - https://github.com/pytorch/pytorch/issues/16963

    :param a : Matrix"""

    a_norm_1d = torch.linalg.matrix_norm(a,ord=1)
    a_norm_inf = torch.linalg.matrix_norm(a,ord=float('inf'))

    alpha = 1/(a_norm_1d*a_norm_inf)
    N_0 = alpha*a.T

    N_t = N_0
    In = torch.eye(a.shape[0])

    for t in range(iterations):
        N_t_plus_1 = N_t@(3*In - a@N_t@(3*In - a@N_t))
        N_t = N_t_plus_1

    return N_t_plus_1


print("Chebyshev inverse")
i_c = chebyshev_inverse(a,iterations=10)

b = a.expand(4,3,3)

i = torch.linalg.inv(b)
print("Torch's inverse")
print(i)

def chebyshev_inverse_3d(a, iterations=10):
    """Computes an approximation of the inverse of a matrix
    :param a : Matrix"""

    a_norm_1d = torch.linalg.matrix_norm(a, ord=1)

    a_norm_inf = torch.linalg.matrix_norm(a, ord=float('inf'))


    alpha = 1 / (a_norm_1d * a_norm_inf)

    N_0 = alpha[:,None,None] * a.permute(0,2,1)

    N_t = N_0
    In = torch.eye(a.shape[1]).expand(a.shape[0],a.shape[1],a.shape[2])

    for t in range(iterations):

        N_t_plus_1 = torch.matmul(N_t , (3 * In - torch.matmul(torch.matmul(a,N_t) , (3*In - torch.matmul(a,N_t)))))
        N_t = N_t_plus_1

    return N_t_plus_1


i3d= chebyshev_inverse_3d(b)


print(i3d)