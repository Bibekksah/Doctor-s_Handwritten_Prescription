def classify_confidence(confidence):

    confidence = float(confidence)

    if confidence >= 0.80:
        return {
            "level": "high",
            "requires_verification": False,
        }

    if confidence >= 0.50:
        return {
            "level": "medium",
            "requires_verification": True,
        }

    return {
        "level": "low",
        "requires_verification": True,
    }