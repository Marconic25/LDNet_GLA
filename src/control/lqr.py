import numpy as np
import tensorflow as tf
from scipy.linalg import solve_discrete_are
from structural.smd import M_WING, M_FLAP, I_WING, I_FLAP_EA, D_H, D_ALPHA, K_H, K_ALPHA, _D_X


def compute_jacobians(aero_model, x_trim, z_trim, U_INF, DT, CL_trim=0.0, CM_trim=0.0):
    """
    Linearize the augmented system at trim via tf.GradientTape.

    Augmented state: ξ = [h, hd, α, αd, z]  (dim 5)
    Input:           u = δ                    (dim 1)
    Outputs:         y = [ΔC_L, ΔC_M]        (dim 2, deviations from trim)

    Returns:
        A_d  (5×5), B_d  (5×1), C_y  (2×5), D_y  (2×1), B_w  (5,)
    """
    q_dyn = 0.5 * 1.225 * U_INF**2 * 0.05
    CL_tr = tf.constant(CL_trim, dtype=tf.float64)
    CM_tr = tf.constant(CM_trim, dtype=tf.float64)

    M_hh = float(M_WING + M_FLAP)
    M_aa = float(I_WING + I_FLAP_EA)
    M_ha = float(M_FLAP * _D_X)
    det  = M_hh * M_aa - M_ha**2
    dH   = float(D_H); kH = float(K_H)
    dA   = float(D_ALPHA); kA = float(K_ALPHA)

    def rk4_tf(x, dFy, dMz): #RK4 in tensorflow to get gradients of next state w.r.t. current state and input
        DT_tf = tf.constant(DT, dtype=tf.float64)
        def rhs(s): #accelerations given current state and input
            RHS_h = -dFy - dH*s[1] - kH*s[0]
            RHS_a =  dMz - dA*s[3] - kA*s[2]
            return tf.stack([s[1],
                             (M_aa*RHS_h - M_ha*RHS_a) / det,
                             s[3],
                             (M_hh*RHS_a - M_ha*RHS_h) / det])
        k1 = rhs(x); k2 = rhs(x + 0.5*DT_tf*k1)
        k3 = rhs(x + 0.5*DT_tf*k2); k4 = rhs(x + DT_tf*k3)
        return x + (DT_tf/6.0)*(k1 + 2*k2 + 2*k3 + k4)

#Define TF variables for state, input, and gust disturbance (W)
    xi_var = tf.Variable(
        np.concatenate([x_trim, z_trim]).astype(np.float64), dtype=tf.float64)
    u_var  = tf.Variable([0.0], dtype=tf.float64)
    W_var  = tf.Variable(0.0,   dtype=tf.float64)

    with tf.GradientTape(persistent=True) as tape:
        h = xi_var[0]; hd = xi_var[1]
        a = xi_var[2]; ad = xi_var[3]
        z = xi_var[4:5]
        delta = u_var[0]

        #single forward pass starting from the trim point
        z_new, C_L, C_M = aero_model.step_tf(
            z, h, hd, a, ad, delta, W_var,
            tf.constant(U_INF, dtype=tf.float64), DT)

        #incremental forces w.r.t. trim (for output y)
        dFy    = tf.constant(q_dyn, dtype=tf.float64) * (C_L - CL_tr)
        dMz    = tf.constant(q_dyn, dtype=tf.float64) * (C_M - CM_tr)
        x_new  = rk4_tf(xi_var[:4], dFy, dMz) #next state of the structural system given current state and input (without z dynamics))
        xi_new = tf.concat([x_new, z_new], axis=0) #Full next state of the augmented system (including z dynamics)
        y_out  = tf.stack([C_L - CL_tr, C_M - CM_tr]) #Output is the deviation of CL and CM from their trim values, which are the quantities we want to regulate to zero with LQR

    #tape recorded all the operations in the forward pass. now backprop to get the Jacobians of next state and output w.r.t. current state, input, and gust disturbance
    A_d = tape.jacobian(xi_new, xi_var).numpy()   # (5, 5)
    B_d = tape.jacobian(xi_new, u_var).numpy()    # (5, 1)
    C_y = tape.jacobian(y_out,  xi_var).numpy()   # (2, 5)
    D_y = tape.jacobian(y_out,  u_var).numpy()    # (2, 1)
    B_w = tape.jacobian(xi_new, W_var).numpy()    # (5,)
    del tape
    return A_d, B_d, C_y, D_y, B_w


class LQRController:
    """
    LQR on the augmented system ξ = [h, hd, α, αd, z, W_hat] (dim 6).

    W model: W_{k+1} = lambda_w * W_k  (exponential decay).
    The DARE automatically computes the optimal feed-forward gain on W.

    At runtime: δ = -K @ [ξ_struct - ξ_trim; W_hat]
    """

    def __init__(self, aero_model, U_INF, DT, x_trim, z_trim,
                 Q_lqr, R_lqr, CL_trim=0.0, CM_trim=0.0,
                 Q_y=None, Q_w=0.0, lambda_w=0.98, delta_max=20.0,
                 precomputed_jacobians=None):
        self.x_trim    = x_trim.copy()
        self.z_trim    = z_trim.copy()
        self.delta_max = delta_max

        if precomputed_jacobians is not None:
            A_d, B_d, C_y, D_y, B_w = precomputed_jacobians
        else:
            print("  Computing Jacobians at trim...")
            A_d, B_d, C_y, D_y, B_w = compute_jacobians(
                aero_model, x_trim, z_trim, U_INF, DT, CL_trim, CM_trim)

        # Augment system with W state: ξ_aug = [ξ(5), W(1)]
        # ξ_{k+1} = A_d @ ξ_k + B_d @ u + B_w * W_k
        # W_{k+1}  = lambda_w * W_k
        A_aug = np.block([[A_d,              B_w.reshape(5, 1)       ],
                          [np.zeros((1, 5)), np.array([[lambda_w]])  ]])  # (6, 6)
        B_aug = np.vstack([B_d, [[0.0]]])                             # (6, 1)

        eigs_ol = np.linalg.eigvals(A_aug)
        print(f"  Open-loop eigenvalues: {np.abs(eigs_ol).round(4)}")

        # Extend Q to 6×6 (add Q_w weight on W state)
        Q6 = np.zeros((6, 6))
        Q6[:5, :5] = Q_lqr
        Q6[5, 5]   = Q_w

        if Q_y is not None:
            # Extend C_y to include W column (W doesn't directly affect output at trim)
            C_y6 = np.hstack([C_y, np.zeros((2, 1))])
            D_y6 = D_y
            Q6   = Q6 + C_y6.T @ Q_y @ C_y6
            R_aug = R_lqr + D_y6.T @ Q_y @ D_y6
        else:
            R_aug = R_lqr

        print("  Solving DARE (6×6)...")
        P      = solve_discrete_are(A_aug, B_aug, Q6, R_aug)
        Rinv   = np.linalg.inv(R_aug + B_aug.T @ P @ B_aug)
        self.K = Rinv @ (B_aug.T @ P @ A_aug)   # (1, 6)

        eigs_cl = np.linalg.eigvals(A_aug - B_aug @ self.K)
        print(f"  Closed-loop eigenvalues: {np.abs(eigs_cl).round(4)}")
        print(f"  LQR gain K = {self.K.round(4)}")
        print(f"  K[5] (W feed-forward) = {self.K[0,5]:.4f}")

    # Controller called at each time step with current state estimates and gust estimate W_hat
    def solve(self, x_hat, z_hat, W_hat=0.0):
        # Form augmented state ξ = [x_hat - x_trim; z_hat - z_trim; W_hat]
        xi = np.concatenate([x_hat - self.x_trim,
                             z_hat  - self.z_trim,
                             [W_hat]])             # (6,)
        delta = float(-(self.K @ xi)[0]) #The gain operates on the deviation of the current state from trim, and on the gust estimate W_hat, to compute the control input delta. The LQR gain K is designed to minimize deviations in both state and output, while also accounting for the effect of gusts through W_hat.
        return np.clip(delta, -self.delta_max, self.delta_max)
