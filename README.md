# Sampling from manifold-restricted Gaussians

Python implementation of algorithm from [[A. J. K. Chua, Sampling from manifold-restricted distributions using tangent bundle projections, Stat. Comput., in press]](https://doi.org/10.1007/s11222-019-09907-8). Contains iteration class with documentation and example usage.

**<https://arxiv.org/abs/1811.05494>**

---

## algorithm.iteration

### METHODS

**\__init\__**(*betastar, minisize=100, epsilon=1e-2, bounds=None, autocompact=True, replace=True*)

**\__call\__**(*theta, alpha, jacobian, hessian=None*)

### ARGUMENTS

**betastar**: *float array with shape (d,)*  
\- beta_* from Eq. (25) in manuscript

**minisize**: *int, optional*  
\- size of mini-distribution (denoted m in manuscript)

**epsilon**: *float, optional*  
\- tuning parameter for overall spread of mini-distribution

**bounds**: *float array with shape (s,2), optional*  
\- sampling limits, i.e., [[min(theta^1),max(theta^1)],...,[min(theta^s),max(theta^s)]]  
\- if not provided, constant compactness is used

**autocompact**: *bool, optional*  
\- set false to force use of constant compactness

**replace**: *bool, optional*  
\- set false to retain generated points outside sampling limits

**theta**: *float array with shape (s,)*  
\- base point in sample space (denoted theta_i in manuscript)

**alpha**: *float array with shape (d,)*  
\- map evaluated at base point (denoted alpha_i in manuscript)

**jacobian**: *float array with shape (d,s)*  
\- first derivative of map evaluated at base point (denoted J_i in manuscript)

**hessian**: *float array with shape (d,s,s), optional*  
\- second derivative of map evaluated at base point (denoted H_i in manuscript)  
\- if not provided, constant or metric-only compactness is used

### ATTRIBUTES

**samples**: *float array with shape (m,s)*  
\- generated points in sample space  
\- theta_ij from Eq. (13) in manuscript

**logweights**: *float array with shape (m,)*  
\- natural logarithm of sample weights  
\- w_ij from Eqs (30), (41), (42) in manuscript

**metric**: *float array with shape (s,s)*  
\- (F_I)_i from Eq. (3) in manuscript

**pseudoinverse**: *float array with shape (s,d)*  
\- (J^+)_i from Eq. (9) in manuscript

**metricterm**: *float*  
\- (lambda^2)_i from Eqs (37), (38) in manuscript

**second**: *float array with shape (s,s)*  
\- (F_II)_i from Eqs (6), (7) in manuscript

**curvatureterm**: *float*  
\- kappa_i from Eqs (33)-(35) in manuscript

**compactness**: *float*  
\- c_i from Eq. (40) in manuscript

**beta**: *float array with shape (m,d)*  
\- beta_ij from Eq. (5) in manuscript

**pushforward**: *float array with shape (m,d)*  
\- (beta^perp)_ij from Eq. (14) in manuscript

**timing**: *float array with shape (6,)*  
\- timing of Steps 2-6 and boundary corrections in manuscript