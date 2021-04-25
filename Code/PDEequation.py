import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None
    
    def setTotal_time(self,totaTime):
        self.total_time=totaTime
    
    
    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError



class PricingBS(Equation):
    def __init__(self, eqn_config,stock_config):
        super(PricingBS, self).__init__(eqn_config)
        self.strike_price = stock_config.strike_price
        s_min_interest = stock_config.s_min_interest
        s_max_interest = stock_config.s_max_interest
        volatility_min = stock_config.volatility_min
        volatility_max = stock_config.volatility_max
        correlation_min = stock_config.correlation_min
        correlation_max = stock_config.correlation_max
        self.x_init=stock_config.x_init
        self.sigma=np.random.uniform(volatility_min, volatility_max, self.dim)
        self.cov=np.ones((self.dim,self.dim))
        for i in range(self.dim):
            self.cov[i,i]=1
            for j in range(i+1,self.dim):
                self.cov[i,j]=np.random.uniform(correlation_min, correlation_max, 1)
                self.cov[j,i]=self.cov[i,j]
        self.r= stock_config.riskfree_rate_eval

    def getCov(self):
        return self.cov
    
    def setCov(self,Cov):
        self.cov=Cov
       
    def setRate(self,r):
        self.r=r

    
    def setX_init(self,Strike=100,x_init=100):
        self.strike_price = Strike
        self.x_init=x_init


    def sample(self, num_sample):
        dw_sample= np.zeros([num_sample, self.dim, self.num_time_interval])
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t)
        for i in range(self.num_time_interval):
            dw_sample[:, :, i ] = normal.rvs(cov=self.cov,size=[num_sample],random_state=i) * self.sqrt_delta_t
            x_sample[:, :, i + 1] = factor *( np.exp(self.sigma * dw_sample[:, :, i]) * x_sample[:, :, i])
        return dw_sample, x_sample
    
    def sample2(self, num_sample):
        dw_sample= np.zeros([num_sample, self.dim, self.num_time_interval])
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            dw_sample[:, :, i ] = normal.rvs(cov=self.cov,size=[num_sample],random_state=i) * self.sqrt_delta_t
            x_sample[:, :, i + 1] = x_sample[:, :, i]*(1+self.r*self.delta_t*np.ones(self.dim) + self.sigma*dw_sample[:, :, i])
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.r * y #+tf.reduce_sum(z *((self.sigma**2)/2), 1, keepdims=True)

    def g_tf(self, t, x): # payof basket call option
        temp = tf.reduce_mean(x, 1, keepdims=True)
        return tf.maximum(temp - self.strike_price, 0)

