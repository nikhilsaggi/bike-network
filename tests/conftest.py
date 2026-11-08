def lion_row(seg, u, v, traf="T", **kw):
    """One raw LION row with LION's own column names and space-padded codes."""
    row = {
        "SegmentID": f"{seg:07d}",
        "Street": kw.pop("street", f"STREET {seg}"),
        "FeatureTyp": "0",
        "SegmentTyp": "U",
        "RB_Layer": "B",
        "NonPed": " ",
        "TrafDir": traf,
        "Status": "2",
        "RW_TYPE": " 1",
        "NodeIDFrom": f"{u:07d}",
        "NodeIDTo": f"{v:07d}",
        "BikeLane": "  ",
        "BIKE_TRAFDIR": "  ",
        "POSTED_SPEED": "  ",
        "Number_Travel_Lanes": " 2",
        "Number_Park_Lanes": " 2",
        "TRUCK_ROUTE_TYPE": " ",
        "LBoro": 1,
        "SHAPE_Length": kw.pop("feet", 1000.0 / 0.3048),  # 1 km by default
    }
    row.update(kw)
    return row


def dot_row(seg, ft=None, tf=None, status="Current", **kw):
    return {"segmentid": str(seg), "status": status, "ft_facilit": ft, "tf_facilit": tf, **kw}
