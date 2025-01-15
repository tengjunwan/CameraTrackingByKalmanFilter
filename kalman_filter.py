import numpy as np


class KalmanFilter():
    def __init__(self, dt, meas_std=0.01, acc_std=20):
        """
        Args:
            dt: prediction time interval;
            meas_std: measure standard deviation, it describes the uncertainty of 
                measurement;
            acc_std: accerlation standard deviation, it describes 
                the uncertainty of motion prediction;

        distribution:
            bel_bar(xt) ~ N(xt_bar, Σt_bar)
            bel(xt) ~ N(xt, Σt)
        predict: 
            Xt = A * Xt-1 + B*ut + εt
            bel(xt-1) -> bel_bar(xt)
            var(εt) = R
        correct: 
            Zt = C * Xt + δt
            bel(xt-1)_bar -> bel(xt)
            var(δt) = Q
        
        """
        self.dt = dt

        # bel distribution(initial state)
        self.x = np.array([[0], 
                           [0],
                           [0],
                           [0]], dtype=np.float32)
        self.sigma = np.eye(self.x.shape[0], dtype=np.float32) * 50

        # bel bar distribution
        self.x_bar = self.x.copy()
        self.sigma_bar = self.sigma.copy()


        # predict
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.B = np.array([[-0.5*dt**2, 0],
                           [0, -0.5*dt**2],
                           [-dt, 0],
                           [0, -dt]], dtype=np.float32)
        self.R = np.array([[0.25*dt**4, 0, 0.5*dt**3, 0],
                           [0, 0.25*dt**4, 0, 0.5*dt**3],
                           [0.5*dt**3, 0, dt**2, 0],
                           [0, 0.5*dt**3, 0, dt**2]], dtype=np.float32) \
                            * acc_std**2

        # correct
        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.array([[1, 0],
                           [0, 1]], dtype=np.float32) * meas_std**2
        
    def predict(self, ut):
        # u_(t) = [[ax_(t)], [ay_(t)]]
        if ut.shape != (2, 1):
            ut = ut.reshape(2, 1)
        # Prediction mean: x_(t)_bar = A*x_(t-1) + B*u_(t)   
        self.x_bar = self.A @ self.x + self.B @ ut

        # Prediction covariance: Σ_bar = A*Σ*A' + R         
        self.sigma_bar = self.A @ self.sigma @ self.A.T + self.R
        return self.x_bar[0:2]  # only return x, y 

    def correct(self, zt):
        # z_(t) = [[rel_x], [rel_y]]
        if zt is None:  # no measurement 
            self.x = self.x_bar
            self.sigma = self.sigma_bar
        else:
            if zt.shape != (2, 1):
                zt = zt.reshape(2, 1)
            # term = C*Σ_bar*C'+ Q
            term = self.C @ self.sigma_bar @ self.C.T + self.Q
    
            # Kalman Gain
            # K = Σ_bar*C'*inv(C*Σ_bar*C'+ Q)
            # K = np.dot(np.dot(self.sigma_bar, self.C.T), np.linalg.inv(term))  
            K = self.sigma_bar @ self.C.T @ np.linalg.inv(term)
            
            # x = x_bar + K*(z - C*x_bar) 
            self.x = self.x_bar \
                            + np.dot(K, (zt - np.dot(self.C, self.x_bar)))  
            self.x = self.x_bar + K @ (zt - self.C @ self.x_bar)
    
            # Update Σ
            # ∑ = (I - K*C)∑_bar
            I = np.eye(self.C.shape[1])
            # self.sigma = np.dot((I - np.dot(K, self.C)), self.sigma_bar)   
            self.sigma = (I - K @ self.C) @ self.sigma_bar
    
    def get_position(self):
        return self.x.flatten()[:2]

    def get_velocity(self):
        return self.x.flatten()[2:4]
    
    def get_predcited_position(self):
        return self.x_bar.flatten()[:2]
    
    def get_predcited_velocity(self):
        return self.x_bar.flatten()[2:4]
