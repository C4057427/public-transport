import os, zipfile, glob, xml.etree.ElementTree as ET
import pandas as pd
from .utils import ensure_dir

def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag

def _iterall(root, tag):
    for elem in root.iter():
        if _strip_ns(elem.tag) == tag:
            yield elem

def extract_from_xml(xml_path: str):
    try:
        tree = ET.parse(xml_path); root = tree.getroot()
    except Exception:
        return {"stops": pd.DataFrame(), "journeys": pd.DataFrame(), "patterns": pd.DataFrame()}
    # Stops
    rows_stops = []
    for sp in _iterall(root, "StopPoint"):
        atco=name=lat=lon=None
        for ch in sp:
            tg=_strip_ns(ch.tag)
            if tg in ("AtcoCode","StopPointRef"): atco=(ch.text or "").strip()
            if tg=="Descriptor":
                for s in ch:
                    if _strip_ns(s.tag) in ("CommonName","ShortCommonName"):
                        name=(s.text or "").strip()
            if tg=="Location":
                for s in ch:
                    if _strip_ns(s.tag)=="Latitude": lat=(s.text or "").strip()
                    if _strip_ns(s.tag)=="Longitude": lon=(s.text or "").strip()
        if atco:
            rows_stops.append({"stop_id":atco,"stop_name":name,"lat":lat,"lon":lon})
    stops_df=pd.DataFrame(rows_stops).drop_duplicates()
    # VehicleJourneys（Planning Layer）
    rows_j=[]
    for vj in _iterall(root,"VehicleJourney"):
        line_ref=direction=departure_time=vj_code=jp_ref=None
        for ch in vj:
            tg=_strip_ns(ch.tag)
            if tg in ("LineRef","RouteRef"): line_ref=(ch.text or "").strip()
            if tg in ("Direction","DirectionRef"): direction=(ch.text or "").strip()
            if tg=="DepartureTime": departure_time=(ch.text or "").strip()
            if tg in ("VehicleJourneyCode","VehicleJourneyIdentifier","JourneyCode"): vj_code=(ch.text or "").strip()
            if tg in ("JourneyPatternRef","JourneyPattern"): jp_ref=(ch.text or "").strip()
        rows_j.append({"line_ref":line_ref,"direction":direction,"departure_time":departure_time,
                       "vehicle_journey_code":vj_code,"journey_pattern_ref":jp_ref})
    journeys_df=pd.DataFrame(rows_j).drop_duplicates()
    # JourneyPatternSection Station sequence (if any)
    rows_p=[]
    for jp in _iterall(root,"JourneyPatternSection"):
        sec_id=jp.attrib.get("id") or jp.attrib.get("Id") or ""
        order=0; to_ref=None
        for tl in _iterall(jp,"JourneyPatternTimingLink"):
            from_ref=None
            for sub in tl:
                st=_strip_ns(sub.tag)
                if st=="From":
                    for s in sub:
                        if _strip_ns(s.tag) in ("StopPointRef","StopPoint"): from_ref=(s.text or "").strip()
                if st=="To":
                    for s in sub:
                        if _strip_ns(s.tag) in ("StopPointRef","StopPoint"): to_ref=(s.text or "").strip()
            if from_ref:
                rows_p.append({"section_id":sec_id,"order":order,"stop_id":from_ref}); order+=1
        if order>0 and to_ref:
            rows_p.append({"section_id":sec_id,"order":order,"stop_id":to_ref})
    patterns_df=pd.DataFrame(rows_p).drop_duplicates()
    return {"stops":stops_df,"journeys":journeys_df,"patterns":patterns_df}

def build_timetable(timetable_zip, timetable_dir, out_dir):
    ensure_dir(out_dir)
    # Automatic decompression
    if timetable_zip and os.path.exists(timetable_zip):
        extract_dir=os.path.join(os.path.dirname(out_dir),"timetable_extracted"); ensure_dir(extract_dir)
        with zipfile.ZipFile(timetable_zip,'r') as z: z.extractall(extract_dir)
        timetable_dir=extract_dir
    if not timetable_dir or not os.path.exists(timetable_dir):
        return {}
    xmls=glob.glob(os.path.join(timetable_dir,"**","*.xml"), recursive=True)
    if not xmls: return {}
    stops_all=[]; journeys_all=[]; patterns_all=[]

    for xp in xmls:
        d=extract_from_xml(xp)
        if len(d["stops"]): stops_all.append(d["stops"])
        if len(d["journeys"]): journeys_all.append(d["journeys"])
        if len(d["patterns"]): patterns_all.append(d["patterns"])
    stops_df=pd.concat(stops_all, ignore_index=True).drop_duplicates() if stops_all else pd.DataFrame()
    journeys_df=pd.concat(journeys_all, ignore_index=True).drop_duplicates() if journeys_all else pd.DataFrame()
    patterns_df=pd.concat(patterns_all, ignore_index=True).drop_duplicates() if patterns_all else pd.DataFrame()
    out_stops=os.path.join(out_dir,"stops.parquet")
    out_j=os.path.join(out_dir,"journeys.parquet")
    out_p=os.path.join(out_dir,"patterns.parquet")
    if len(stops_df): stops_df.to_parquet(out_stops, index=False)
    if len(journeys_df): journeys_df.to_parquet(out_j, index=False)
    if len(patterns_df): patterns_df.to_parquet(out_p, index=False)
    return {"stops":out_stops,"journeys":out_j,"patterns":out_p}
