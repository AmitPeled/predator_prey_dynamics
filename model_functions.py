def f_na(C, N, c):  # focus drawn by alerting individual
    if 0 == C * (c - 1) + N:
        raise ValueError(f'C * (c - 1) + N cannot be 0 when calculating f_na.')
    return 1 / (C * (c - 1) + N)


def f_a(C, N, c):
    return c * f_na(C, N, c)


def p_a(C, N, p_prime, c):
    if C == 0:
        raise ValueError(f'C cannot be 0 when calculating p_a. {C=}')
    return 1 - p_prime * f_a(C, N, c)


def p_na(C, N, p, p_prime, c):
    if N == C:
        raise ValueError(f'N cannot be equal to C when calculating p_na. {N=} {C=}')
    p_used = p_prime
    if C == 0:
        p_used = p
    return 1 - p_used * f_na(C, N, c)


# Includes a continuous decrease of p as function of C - No Braess paradox in such environment.
def p_a_c(C, N, p, c, a):
    prob = p - a*C/N
    return 1 - prob * f_a(C, N, c)


def p_na_c(C, N, p, c, a):
    prob = p - a*C/N
    return 1 - prob * f_na(C, N, c)

    
def p_ka(r, C, N, p, p_prime, c):
    if C == 0:
        raise ValueError(f'C cannot be 0 when calculating p_a. {C=}')
    
    if C == 1:
        k_a = r * (N-C)*(p_na(C, N, p, p_prime, c) - p_na(C-1, N, p, p_prime, c))
    elif C == N:
        k_a = r * (C-1)*(p_a(C, N, p_prime, c) - p_a(C-1, N, p_prime, c))
    else:
        k_a = r * ((C-1)*(p_a(C, N, p_prime, c) - p_a(C-1, N, p_prime, c)) + (N-C)*(p_na(C, N, p, p_prime, c) - p_na(C-1, N, p, p_prime, c)))
    
    return p_a(C, N, p_prime, c) + k_a
    