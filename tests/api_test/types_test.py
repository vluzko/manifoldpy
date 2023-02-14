from manifoldpy import api


def test_weak_unstructure():
    json_dict = {
        "id": "6qEWrk0Af7eWupuSWxQm",
        "creatorUsername": "ampdot",
        "creatorName": "ampdot",
        "createdTime": 1649943317395,
        "question": "Will Elon Musk own more than 90% of Twitter before June 1st?",
        "tags": ["twitter", "elon", "Technology"],
        "url": "https://manifold.markets/ampdot/will-elon-musk-own-more-than-90-of",
        "pool": {"NO": 22.406104954208892, "YES": 1322.035893303773},
        "volume": 3227.934376240527,
        "volume24Hours": 0,
        "outcomeType": "BINARY",
        "mechanism": "cpmm-1",
        "isResolved": True,
        "resolutionProbability": 0.014263247555638424,
        "p": 0.46055501243405605,
        "totalLiquidity": 124.13960497195119,
        "closeTime": 1654142340000,
        "creatorId": "oEpXdWv0VgO5CIyBLQWaXx0Zsxr2",
        "lastUpdatedTime": 1653891620553,
        "creatorAvatarUrl": "https://firebasestorage.googleapis.com/v0/b/mantic-markets.appspot.com/o/user-images%2FJamesLu%2Fblank-profile-picture-973460_1280.webp?alt=media&token=92a87b28-57c7-40cf-8b80-92555775b47b",
        "resolution": "NO",
        "resolutionTime": 1654151280445,
        "min": None,
        "max": None,
        "isLogScale": None,
        "description": "",
        "textDescription": "",
        "probability": 0.014263247555638422,
        "bets": None,
        "comments": None,
    }
    m = api.Market.from_json(json_dict)
    recon_json = api.weak_unstructure(m)
    assert recon_json == json_dict


def test_weak_unstructure_nested():
    b = api.get_bets(limit=1)
    m = api.get_markets(limit=1)[0]
    m.bets = b
    m.comments = []

    as_json = api.weak_unstructure(m)
    assert len(as_json["bets"]) == 1
    b2 = as_json["bets"][0]
    assert isinstance(b2, dict)


def test_weak_structure():
    json_dict = {
        "id": "6qEWrk0Af7eWupuSWxQm",
        "creatorUsername": "ampdot",
        "creatorName": "ampdot",
        "createdTime": 1649943317395,
        "question": "Will Elon Musk own more than 90% of Twitter before June 1st?",
        "tags": ["twitter", "elon", "Technology"],
        "url": "https://manifold.markets/ampdot/will-elon-musk-own-more-than-90-of",
        "pool": {"NO": 22.406104954208892, "YES": 1322.035893303773},
        "volume": 3227.934376240527,
        "volume24Hours": 0,
        "outcomeType": "BINARY",
        "mechanism": "cpmm-1",
        "isResolved": True,
        "resolutionProbability": 0.014263247555638424,
        "p": 0.46055501243405605,
        "totalLiquidity": 124.13960497195119,
        "closeTime": 1654142340000,
        "creatorId": "oEpXdWv0VgO5CIyBLQWaXx0Zsxr2",
        "lastUpdatedTime": 1653891620553,
        "creatorAvatarUrl": "https://firebasestorage.googleapis.com/v0/b/mantic-markets.appspot.com/o/user-images%2FJamesLu%2Fblank-profile-picture-973460_1280.webp?alt=media&token=92a87b28-57c7-40cf-8b80-92555775b47b",
        "resolution": "NO",
        "resolutionTime": 1654151280445,
        "min": None,
        "max": None,
        "isLogScale": None,
        "description": "",
        "textDescription": "",
        "probability": 0.014263247555638422,
        "bets": [],
        "comments": [],
    }
    m = api.Market.from_json(json_dict)
