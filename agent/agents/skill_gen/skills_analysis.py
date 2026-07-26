import json
import glob
import os
from collections import defaultdict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ============================================================
# ENUM DEFINITIONS (MATCHES NEW CLASSIFIER PROMPT)
# ============================================================

ALL_ENUMS = {
    "propulsion_mechanism": [
        "CYCLIC_LEG_SWEEP",
        "CONTACT_REDIRECTION",
        "BASE_TRAJECTORY_DRIVEN",
        "IMPULSE_GENERATION",
        "NO_PROPULSION",
    ],
    "contact_role_topology": [
        "FIXED_SUPPORT",
        "ALTERNATING_SUPPORT",
        "CONTINUOUS_REDIRECTING",
        "TRANSIENT_CONTACT",
        "NO_CONTACT",
    ],
    "phase_function": [
        "CONTACT_TIMING",
        "LIMB_SHAPE_EVOLUTION",
        "BASE_ORIENTATION_CHANGE",
        "BASE_POSITION_PROGRESS",
        "EVENT_EXECUTION",
        "AERIAL_ARC",
    ],
    "locomotion_geometry": [
        "FORWARD",
        "BACKWARD",
        "LATERAL",
        "DIAGONAL",
        "CURVED",
        "ORBITAL",
    ],
    "inter_limb_coordination": [
        "SYNCHRONOUS_ALL_LIMBS",
        "FRONT_REAR_COUPLED",
        "LEFT_RIGHT_COUPLED",
        "DIAGONAL_PAIRING",
        "SEQUENTIAL_WAVE",
        "ALTERNATING_SUPPORT_GROUPS",
    ],
    "base_kinematics": [
        "TRANSLATION",
        "YAW",
        "PITCH",
        "ROLL",
        "HEIGHT",
    ],
    "locomotion_mode": [
        "LEGS_WITH_ACTIVE_WHEELS",
        "LEGS_WITH_PASSIVE_WHEELS",
        "AERIAL",
    ],
}

# ============================================================
# VECTOR ENCODING
# ============================================================

SINGLE_CHOICE_FIELDS = ["propulsion_mechanism", "contact_role_topology", "phase_function"]

def skill_to_vector(skill_cls: dict, enums=ALL_ENUMS) -> np.ndarray:
    """
    Convert a single classified skill into a fixed-length binary vector.
    Handles single- and multi-choice fields automatically.
    """
    vec = []
    for field, choices in enums.items():
        if field in SINGLE_CHOICE_FIELDS:
            value = skill_cls[field]
            vec.extend([1 if value == e else 0 for e in choices])
        else:
            values = set(skill_cls.get(field, []))
            vec.extend([1 if e in values else 0 for e in choices])
    return np.array(vec, dtype=np.float32)

def build_feature_matrix(skill_classifications, enums=ALL_ENUMS):
    """
    Convert all skills into a stacked feature matrix.
    """
    vectors = [skill_to_vector(s, enums=enums) for s in skill_classifications]
    return np.stack(vectors, axis=0)

# ============================================================
# LOADING UTILITIES
# ============================================================

def load_classification_files(path_pattern):
    """
    Load multiple JSON classification files. Inject skill_id from filename.
    """
    skills = []
    for path in glob.glob(path_pattern):
        with open(path, "r") as f:
            data = json.load(f)
        skill_id = os.path.splitext(os.path.basename(path))[0]
        data["skill_id"] = skill_id
        skills.append(data)
    return skills

# ============================================================
# CLUSTERING
# ============================================================

def cluster_skills(skill_classifications, n_clusters=8, linkage="average", enums=ALL_ENUMS):
    """
    Global clustering using all categories combined.
    """
    X = build_feature_matrix(skill_classifications, enums=enums)
    D = cosine_distances(X)
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage,
    )
    labels = clusterer.fit_predict(D)
    return labels, X

def group_by_cluster(skill_classifications, labels):
    """
    Map cluster IDs to lists of skill IDs.
    """
    clusters = defaultdict(list)
    for skill, label in zip(skill_classifications, labels):
        clusters[label].append(skill["skill_id"])
    return clusters

def summarize_cluster(cluster_skills):
    """
    Summarize counts of enums in each cluster.
    """
    summary = defaultdict(lambda: defaultdict(int))
    for skill in cluster_skills:
        for field, choices in ALL_ENUMS.items():
            if field in SINGLE_CHOICE_FIELDS:
                summary[field][skill[field]] += 1
            else:
                for v in skill.get(field, []):
                    summary[field][v] += 1
    return dict(summary)

def cluster_prototypes(X, labels):
    """
    Return the index of the skill closest to each cluster centroid.
    """
    prototypes = {}
    for cid in set(labels):
        idx = np.where(labels == cid)[0]
        centroid = X[idx].mean(axis=0)
        dists = np.linalg.norm(X[idx] - centroid, axis=1)
        prototypes[cid] = idx[np.argmin(dists)]
    return prototypes

# ============================================================
# PCA VISUALIZATION (GLOBAL ONLY)
# ============================================================

def embed_pca(X, dim=2):
    pca = PCA(n_components=dim)
    return pca.fit_transform(X)

def visualize_pca(X, labels, skill_ids):
    X2 = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", alpha=0.8, s=60)
    prototypes = cluster_prototypes(X, labels)
    for cid, i in prototypes.items():
        plt.text(X2[i, 0], X2[i, 1], skill_ids[i], fontsize=10, fontweight="bold")
    plt.title("Global Skill Clusters (PCA)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.show()

# ============================================================
# PER-CATEGORY ENUM SUMMARY
# ============================================================

def summarize_category(skills, category):
    """
    Count unique combinations of enums in a category.
    - Single-choice fields: count as usual.
    - Multi-choice fields: count unique sets of enums.
    - Empty or missing fields are labeled as 'NONE'.
    """
    counts = defaultdict(int)
    skill_map = defaultdict(list)  # map combination -> skill_ids

    for skill in skills:
        value = skill.get(category, None)

        # Handle empty or missing values
        if value is None or (isinstance(value, list) and len(value) == 0):
            key = ("NONE",)
        elif isinstance(value, list):
            key = tuple(sorted(value))
        else:
            key = (value,)

        counts[key] += 1
        skill_map[key].append(skill["skill_id"])

    return counts, skill_map

def fix_locomotion_geometry(skills):
    """
    Re-classify skills where locomotion_geometry is empty (NONE).
    Rules:
    - If skill has aerial motion: locomotion_geometry -> ('AERIAL',)
    - If skill has base rotation but no translation: locomotion_geometry -> ('IN_PLACE_DYNAMIC',)
    - Else keep as ('NONE',)
    """
    for skill in skills:
        lg = skill.get("locomotion_geometry", [])
        if lg and len(lg) > 0:
            continue  # already has geometry, nothing to do

        # empty/missing -> check other cues
        base = set(skill.get("base_kinematics", []))
        phase = set(skill.get("phase_function", []))

        if "BASE_HEIGHT_MODULATION" in base or "AERIAL_ARC" in phase:
            skill["locomotion_geometry"] = ["AERIAL"]
        elif any(b in base for b in ["YAW", "ROLL", "PITCH"]) and "TRANSLATION" not in base:
            skill["locomotion_geometry"] = ["IN_PLACE_DYNAMIC"]
        else:
            skill["locomotion_geometry"] = ["NONE"]

    return skills


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------
    CLASSIFICATION_JSON_GLOB = os.environ.get("CLASSIFICATION_GLOB", "data/skills/skills_v1/analysis/*.json")
    NUM_CLUSTERS_GLOBAL = 5

    # --------------------------------------------------------
    # LOAD SKILLS
    # --------------------------------------------------------
    skill_classifications = load_classification_files(CLASSIFICATION_JSON_GLOB)
    skill_classifications = fix_locomotion_geometry(skill_classifications)
    assert len(skill_classifications) > 0, "No skills loaded."
    skill_ids = [s["skill_id"] for s in skill_classifications]

    # --------------------------------------------------------
    # GLOBAL CLUSTERING
    # --------------------------------------------------------
    labels_global, X_global = cluster_skills(skill_classifications, n_clusters=NUM_CLUSTERS_GLOBAL)
    clusters_global = group_by_cluster(skill_classifications, labels_global)

    print("\n=== GLOBAL CLUSTER ASSIGNMENTS ===")
    for cid, skill_list in clusters_global.items():
        print(f"\nCluster {cid} ({len(skill_list)} skills): {skill_list}")

    print("\n=== GLOBAL CLUSTER SUMMARIES ===")
    for cid, skill_list in clusters_global.items():
        skills = [s for s in skill_classifications if s["skill_id"] in skill_list]
        summary = summarize_cluster(skills)
        print(f"\nCluster {cid} summary:")
        for field, counts in summary.items():
            print(f"  {field}: {dict(counts)}")

    visualize_pca(X_global, labels_global, skill_ids)

    # --------------------------------------------------------
    # PER-CATEGORY ENUM COUNTS
    # --------------------------------------------------------
    print("\n=== PER-CATEGORY ENUM COUNTS ===")
    for category in ALL_ENUMS.keys():
        print(f"\n=== CATEGORY SUMMARY: {category} ===")
        counts, skill_map = summarize_category(skill_classifications, category)
        for combo, cnt in counts.items():
            print(f"  {combo}: {cnt} skills")
        # Optional: print skill IDs for empty combos
        if ("NONE",) in skill_map:
            print(f"    Skills with NONE: {skill_map[('NONE',)]}")
