# utils.py
import random
import json

def sample_value(attr_cfg):
    if attr_cfg is None:
        return None

    dist = attr_cfg.get("distribution")
    if dist == "uniform":
        return random.uniform(attr_cfg["min"], attr_cfg["max"])
    if dist == "normal":
        return random.gauss(attr_cfg["mean"], attr_cfg["std"])
    if "fixed" in attr_cfg:
        return attr_cfg["fixed"]
    if "value" in attr_cfg:
        return attr_cfg["value"]
    if "min" in attr_cfg and "max" in attr_cfg:
        return random.uniform(attr_cfg["min"], attr_cfg["max"])
    return None

def sample_gender(gender_ratio):
    """gender_ratio: {'female':0.6,'male':0.4}"""
    if not gender_ratio:
        return None
    choices = list(gender_ratio.keys())
    weights = list(gender_ratio.values())
    return random.choices(choices, weights=weights, k=1)[0]

def sample_age(age_range):
    if not age_range:
        return None
    return random.randint(age_range[0], age_range[1])

def sample_agent_attributes(agent_type, agents_file="agents.json"):
    with open(agents_file, "r") as f:
        data = json.load(f)

    agent_info = data["agent_types"].get(agent_type)
    if not agent_info:
        raise ValueError(f"Agent type '{agent_type}' not found in {agents_file}")

    attrs = {}

    # attributes
    for attr_name, cfg in agent_info.get("attributes", {}).items():
        attrs[attr_name] = sample_value(cfg)

    dist_cfg = agent_info.get("distribution", {})
    attrs["gender"] = sample_gender(dist_cfg.get("gender_ratio"))
    attrs["age"] = sample_age(dist_cfg.get("age_range"))

    return attrs

# # utils.py
# import random
# import json

# def sample_value(attr_cfg):
#     if attr_cfg is None:
#         return None

#     dist = attr_cfg.get("distribution")
#     if dist == "uniform":
#         return random.uniform(attr_cfg["min"], attr_cfg["max"])
#     if dist == "normal":
#         return random.gauss(attr_cfg["mean"], attr_cfg["std_dev"])
#     if "fixed" in attr_cfg:
#         return attr_cfg["fixed"]
#     if "value" in attr_cfg:
#         return attr_cfg["value"]
#     if "min" in attr_cfg and "max" in attr_cfg:
#         return random.uniform(attr_cfg["min"], attr_cfg["max"])
#     return None

# def sample_gender(gender_ratio):
#     """gender_ratio: {'female':0.6,'male':0.4}"""
#     if not gender_ratio:
#         return None
#     choices = list(gender_ratio.keys())
#     weights = list(gender_ratio.values())
#     return random.choices(choices, weights=weights, k=1)[0]

# def sample_age(age_range):
#     if not age_range:
#         return None
#     return random.randint(age_range[0], age_range[1])

# def sample_agent_attributes(agent_type, agents_file="agents.json"):
#     with open(agents_file, "r") as f:
#         data = json.load(f)

#     agent_info = data["agent_types"].get(agent_type)
#     if not agent_info:
#         raise ValueError(f"Agent type '{agent_type}' not found in {agents_file}")

#     attrs = {}

#     # attributes
#     for attr_name, cfg in agent_info.get("attributes", {}).items():
#         attrs[attr_name] = sample_value(cfg)

#     dist_cfg = agent_info.get("distribution", {})
#     attrs["gender"] = sample_gender(dist_cfg.get("gender_ratio"))
#     attrs["age"] = sample_age(dist_cfg.get("age_range"))

#     return attrs
