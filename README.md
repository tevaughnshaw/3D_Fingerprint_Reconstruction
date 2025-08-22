# Abstract

Reliable and accurate fingerprint recognition is important for biometric systems used in security, authentication, and personal identification. However, current technologies often rely on surface-level imaging, which can be limited by skin conditions, environmental factors, or partial contact.  

This study explores a new, *in-vivo* approach for enhancing the accuracy and robustness of fingerprint analysis by using Optical Coherence Tomography (OCT) scans of fingertip skin and reconstructed 3D meshes of the stratum corneum with associated sweat duct entry and exit points. OCT allows for high-resolution, non-invasive imaging of subsurface skin layers, providing large amounts of volumetric data that can be used to capture internal features, such as sweat ducts and ridge structures.  

With a goal to identify consistent and trackable internal features that could serve as reliable biometric markers, we developed a pipeline for interactive region-of-interest selection using a bounding box, mesh cropping, and anatomical segmentation of internal ridge features with an interactive plane widget to ensure a clean dataset. DBSCAN was then applied to group sweat duct points, and a single centroid was calculated for each cluster to reduce the dataset while preserving key spatial information.  

The core of our analysis involved applying geometric metrics to these reduced centroids. We characterized the ridge geometry by examining spatial distribution, surface curvature, and normal vector orientation. Our key findings were:  

- **Mean nearest-neighbor distance** of sweat duct points: **0.1259 mm** (SD = 0.0444 mm)  
- **Average mean curvature**: **5.6783** (SD = 51.9530), statistically significant (*p* = 1.3726 × 10⁻³)  
- **Average normal vector dot product**: **0.7891** (SD = 0.1995), significantly deviating from a perfectly flat surface (*p* = 4.2014 × 10⁻¹²⁷)  

These results provide a quantitative description of the spatial, curvature, and orientation characteristics of internal fingerprint features, offering a strong foundation for future advancements in more reliable biometric systems.


Full paper details [here](https://github.com/tevaughnshaw/MSc_dissertation_paper).