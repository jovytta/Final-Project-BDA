import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

def optimized_clustering(data, n_clusters):
    """Optimized clustering that tests multiple methods and picks the best."""
    features = data.drop('id', axis=1).values
    
    best_score = -1
    best_labels = None
    
    # Method 1: QuantileTransformer + GMM (multiple seeds)
    try:
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        features_qt = StandardScaler().fit_transform(qt.fit_transform(features))
        
        for seed in [42, 123, 456, 789]:
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type='full',
                random_state=seed,
                max_iter=500,
                n_init=15,
                init_params='k-means++'
            )
            labels = gmm.fit_predict(features_qt)
            
            if len(np.unique(labels)) == n_clusters:
                # Use silhouette score to evaluate quality
                from sklearn.metrics import silhouette_score
                sample_size = min(2000, len(features_qt))
                idx = np.random.choice(len(features_qt), sample_size, replace=False)
                score = silhouette_score(features_qt[idx], labels[idx])
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
    except:
        pass
    
    # Method 2: PowerTransformer + GMM (multiple covariance types)
    try:
        pt = PowerTransformer(method='yeo-johnson')
        features_pt = StandardScaler().fit_transform(pt.fit_transform(features))
        
        for cov_type in ['full', 'tied', 'diag']:
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type=cov_type,
                random_state=42,
                max_iter=500,
                n_init=15
            )
            labels = gmm.fit_predict(features_pt)
            
            if len(np.unique(labels)) == n_clusters:
                from sklearn.metrics import silhouette_score
                sample_size = min(2000, len(features_pt))
                idx = np.random.choice(len(features_pt), sample_size, replace=False)
                score = silhouette_score(features_pt[idx], labels[idx])
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
    except:
        pass
    
    # Method 3: Enhanced features + PCA
    try:
        # Create enhanced features
        enhanced_features = np.column_stack([
            features,
            np.mean(features, axis=1),
            np.std(features, axis=1),
        ])
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(features.shape[1] + 2, enhanced_features.shape[1]))
        features_pca = pca.fit_transform(StandardScaler().fit_transform(enhanced_features))
        
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='tied',
            random_state=42,
            max_iter=400,
            n_init=10
        )
        labels = gmm.fit_predict(features_pca)
        
        if len(np.unique(labels)) == n_clusters:
            from sklearn.metrics import silhouette_score
            sample_size = min(2000, len(features_pca))
            idx = np.random.choice(len(features_pca), sample_size, replace=False)
            score = silhouette_score(features_pca[idx], labels[idx])
            
            if score > best_score:
                best_score = score
                best_labels = labels
    except:
        pass
    
    if best_labels is not None:
        return best_labels
    
    # Fallback method
    features_fallback = StandardScaler().fit_transform(
        QuantileTransformer(output_distribution='normal').fit_transform(features)
    )
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gmm.fit_predict(features_fallback)

def main():
    """Main function for God Particle clustering using optimized methods."""
    # Load particle accelerator datasets
    public_data = pd.read_csv('public_data.csv')
    private_data = pd.read_csv('private_data.csv')
    
    # Calculate target clusters using 4n-1 formula
    public_clusters = 4 * (len(public_data.columns) - 1) - 1
    private_clusters = 4 * (len(private_data.columns) - 1) - 1
    
    # Process public dataset
    public_labels = optimized_clustering(public_data, public_clusters)
    
    # Save public results
    public_df = pd.DataFrame({
        'id': public_data['id'],
        'label': public_labels
    })
    public_df.to_csv('public_submission.csv', index=False)
    
    # Process private dataset
    private_labels = optimized_clustering(private_data, private_clusters)
    
    # Save private results
    private_df = pd.DataFrame({
        'id': private_data['id'],
        'label': private_labels
    })
    private_df.to_csv('private_submission.csv', index=False)

if __name__ == "__main__":
    main()