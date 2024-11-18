import numpy as np
from tqdm import trange


class MulticlassHKRRAlgorithm:
    """
    Multiclass version of HKRR Algorithm for multicalibration.
    Handles probabilistic predictions for multiple classes.
    """
    def __init__(self, n_classes, verbose=False):
        """
        Initialize the multiclass HKRR algorithm
        
        Args:
            n_classes (int): Number of classes in the classification problem
            verbose (bool): Whether to print progress information
        """
        self.n_classes = n_classes
        self.v_hat_saved = []
        self.delta_iters = None
        self.subgroup_updated_iters = None
        self.v_updated_iters = None
        self.verbose = verbose

    def fit(self, confs, labels, subgroups, params):
        """
        Multicalibrate Predictions on Training Set
        
        Args:
            confs: Initial confidence scores, shape (n_samples, n_classes)
            labels: True labels, shape (n_samples,)
            subgroups: List of lists where each entry contains indices for a subgroup
            params: Dictionary of hyperparameters
        """
        try:
            self.lmbda = params['lambda']
            self.alpha = params['alpha']
            self.max_iter = params['max_iter']
            self.randomized = params['randomized']
            self.use_oracle = params['use_oracle']
        except KeyError as e:
            raise ValueError(f"Missing parameter: {e}. Please provide all required parameters (lambda, alpha, max_iter, randomized, use_oracle) as a dictionary.")

        # One-hot encode labels
        labels = np.eye(self.n_classes)[labels]

        # Initialize predictions
        p = confs.copy()  # Shape: (n_samples, n_classes)
        n = len(confs)
        alpha = self.alpha
        lmbda = self.lmbda

        # Count iterations
        iter = 0
        delta_iters = []
        subgroup_updated_iters = []
        v_updated_iters = []

        # Get probability intervals and subgroups
        V_range = np.arange(0, 1, lmbda)
        C = [(i, sg) for i, sg in enumerate(subgroups)]

        # Repeat until no updates made
        updated = True
        while updated and iter < self.max_iter:
            if self.verbose:
                print(f"Iteration {iter+1}...")
            updated = False
            iter += 1

            # Track steps
            delta = []
            subgroup_updated = []
            v_updated = []

            # Shuffle if randomized
            if self.randomized:
                np.random.shuffle(C)
                np.random.shuffle(V_range)

            # For each subgroup and class
            for S_idx, S in C:
                if len(S) == 0:
                    continue
                
                for class_idx in range(self.n_classes):
                    for v in V_range:
                        # Find points in subgroup where prediction for current class falls in interval
                        S_v = [i for i in S if ((v < p[i, class_idx] <= v + lmbda) or 
                                              (v == 0 and v <= p[i, class_idx] <= v + lmbda))]

                        # Check minimum size threshold
                        tao = alpha * lmbda * len(S)
                        if len(S_v) < tao:
                            continue

                        # Calculate expected probability for current class
                        v_hat = np.mean(p[S_v, class_idx])

                        if self.use_oracle:
                            r = self.oracle(subset=S_v, v_hat=v_hat, omega=(alpha/4), 
                                          labels=labels[:, class_idx])
                            
                            if r != 100:
                                # Update predictions for current class
                                p[S_v, class_idx] = p[S_v, class_idx] + (r - v_hat)
                                # Ensure predictions stay in simplex
                                p[S_v] = self._project_to_simplex(p[S_v])
                                updated = True

                                delta.append((r - v_hat, class_idx))
                                subgroup_updated.append(S_idx)
                                v_updated.append(v)
                        else:
                            dlta = np.mean(labels[S_v, class_idx]) - v_hat
                            if abs(dlta) < lmbda/10:
                                continue
                                
                            # Update predictions for current class
                            p[S_v, class_idx] = p[S_v, class_idx] + dlta
                            # Ensure predictions stay in simplex
                            p[S_v] = self._project_to_simplex(p[S_v])
                            updated = True

                            delta.append((dlta, class_idx))
                            subgroup_updated.append(S_idx)
                            v_updated.append(v)

                        if self.verbose:
                            print(f"Updated estimates for {len(S_v)} points in subgroup {S_idx}, class {class_idx} with v={v}")

            delta_iters.append(delta)
            subgroup_updated_iters.append(subgroup_updated)
            v_updated_iters.append(v_updated)

            # Save v_hats for current iteration
            self.v_hat_saved.append({})
            for class_idx in range(self.n_classes):
                for v in V_range:
                    v_lmbda = [i for i in range(n) if ((v < p[i, class_idx] <= v + lmbda) or 
                                                      (v == 0 and v <= p[i, class_idx] <= v + lmbda))]
                    
                    if len(v_lmbda) == 0:
                        self.v_hat_saved[iter-1][(v, class_idx)] = -1
                        continue

                    v_hat = np.mean(p[v_lmbda, class_idx])
                    self.v_hat_saved[iter-1][(v, class_idx)] = v_hat

        self.lmbda = lmbda
        self.delta_iters = delta_iters
        self.subgroup_updated_iters = subgroup_updated_iters
        self.v_updated_iters = v_updated_iters

        return p

    def _project_to_simplex(self, v):
        """
        Project a batch of vectors onto the probability simplex.
        Pulled from Algorithm 1 in https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf.
        
        Args:
            v: Array of shape (batch_size, n_classes)
        Returns:
            Projected vectors of same shape
        """
        def _project_single(v):
            # Sort v into u in descending order
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u) - 1
            rho = np.nonzero(u * np.arange(1, len(v) + 1) > cssv)[0][-1]
            theta = cssv[rho] / (rho + 1)
            return np.maximum(v - theta, 0)
        
        return np.array([_project_single(x) for x in v])

    def oracle(self, subset, v_hat, omega, labels):
        """Multiclass version of the SQ oracle"""
        ps = np.mean(labels[subset])
        r = 0
        
        if abs(ps-v_hat) < 2*omega:
            r = 100
        if abs(ps-v_hat) > 4*omega:
            r = np.random.uniform(0, 1)
        if r != 100:
            r = np.random.uniform(ps-omega, ps+omega)

        return r

    def _circuit_predict(self, f_x, subgroups_containing_x, early_stop=None):
        """
        Adjust test-set predictions with deltas from multicalibration procedure
        
        Args:
            f_x: Initial prediction vector of shape (n_classes,)
            subgroups_containing_x: List of subgroup indices containing this point
        """
        early_stop = early_stop if early_stop else len(self.subgroup_updated_iters)
        mcb_pred = f_x.copy()

        for subgroup_updated, v_updated, delta in zip(
            self.subgroup_updated_iters[:early_stop],
            self.v_updated_iters[:early_stop],
            self.delta_iters[:early_stop]
        ):
            for lvl in range(len(subgroup_updated)):
                if subgroup_updated[lvl] in subgroups_containing_x:
                    v = v_updated[lvl]
                    d, class_idx = delta[lvl]
                    
                    if (v < mcb_pred[class_idx] <= v + self.lmbda) or \
                       (v == 0 and v <= mcb_pred[class_idx] <= v + self.lmbda):
                        mcb_pred[class_idx] = mcb_pred[class_idx] + d
                        mcb_pred = self._project_to_simplex(mcb_pred.reshape(1, -1))[0]

        # Get final prediction from calibration set v_hats
        V_range = np.arange(0, 1, self.lmbda)
        for class_idx in range(self.n_classes):
            for v in V_range:
                if (v < mcb_pred[class_idx] <= v + self.lmbda) or \
                   (v == 0 and v <= mcb_pred[class_idx] <= v + self.lmbda):
                    if self.v_hat_saved[-1].get((v, class_idx), -1) != -1:
                        mcb_pred[class_idx] = self.v_hat_saved[-1][(v, class_idx)]

        return mcb_pred

    def predict(self, f_xs, groups, early_stop=None):
        """
        Predict for multiple test points
        
        Args:
            f_xs: Initial predictions of shape (n_samples, n_classes)
            groups: List of groups assignments
        """
        early_stop = early_stop if early_stop else len(self.subgroup_updated_iters)
        mcb_preds = f_xs.copy()

        if self.verbose:
            print(f"Predicting {len(f_xs)} data points...")
            range_func = trange
        else:
            range_func = range
        
        for i in range_func(len(f_xs)):
            mcb_preds[i] = self._circuit_predict(
                f_xs[i],
                [j for j in range(len(groups)) if i in groups[j]],
                early_stop=early_stop
            )

        return mcb_preds