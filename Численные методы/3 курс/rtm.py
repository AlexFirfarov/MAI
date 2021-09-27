def rtm(m, d):
    size = m.size
    x = [0 for _ in range(0, size)]
    p, q = [], []

    p.append(-m.C[0] / m.B[0])
    q.append(d[0] / m.B[0])

    for i in range(1, size):
        p_i = -m.C[i] / (m.B[i] + m.A[i] * p[i - 1])
        q_i = (d[i] - m.A[i] * q[i - 1]) / (m.B[i] + m.A[i] * p[i - 1])

        p.append(p_i)
        q.append(q_i)

    x[size - 1] = q[size - 1]
    for i in range(size - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x
