import json,re,datetime as dt
from zoneinfo import ZoneInfo
TZ={'nyc':'America/New_York','london':'Europe/London','seoul':'Asia/Seoul','dallas':'America/Chicago','toronto':'America/Toronto','atlanta':'America/New_York','paris':'Europe/Paris','ankara':'Europe/Istanbul','wellington':'Pacific/Auckland','buenos-aires':'America/Argentina/Buenos_Aires','seattle':'America/Los_Angeles','sao-paulo':'America/Sao_Paulo','miami':'America/New_York','chicago':'America/Chicago','lucknow':'Asia/Kolkata','munich':'Europe/Berlin','tel-aviv':'Asia/Jerusalem','tokyo':'Asia/Tokyo','shanghai':'Asia/Shanghai','singapore':'Asia/Singapore','hong-kong':'Asia/Hong_Kong','milan':'Europe/Rome','madrid':'Europe/Madrid','warsaw':'Europe/Warsaw','taipei':'Asia/Taipei','chongqing':'Asia/Shanghai','beijing':'Asia/Shanghai','wuhan':'Asia/Shanghai','shenzhen':'Asia/Shanghai','chengdu':'Asia/Shanghai','denver':'America/Denver','houston':'America/Chicago','los-angeles':'America/Los_Angeles','san-francisco':'America/Los_Angeles','austin':'America/Chicago','istanbul':'Europe/Istanbul','moscow':'Europe/Moscow','mexico-city':'America/Mexico_City','busan':'Asia/Seoul','amsterdam':'Europe/Amsterdam','helsinki':'Europe/Helsinki','kuala-lumpur':'Asia/Kuala_Lumpur','panama':'America/Panama','jeddah':'Asia/Riyadh','cape-town':'Africa/Johannesburg','guangzhou':'Asia/Shanghai','karachi':'Asia/Karachi','manila':'Asia/Manila','qingdao':'Asia/Shanghai','zhengzhou':'Asia/Shanghai','jinan':'Asia/Shanghai','jakarta':'Asia/Jakarta','lagos':'Africa/Lagos'}
def meta(e):
    s=e.get('seriesSlug')
    if not s: return None
    city=s.replace('-daily-weather','')
    if city not in TZ: return None
    d=dt.datetime.fromisoformat(e['endDate'].replace('Z','+00:00')).date()
    tz=ZoneInfo(TZ[city])
    mid=dt.datetime(d.year,d.month,d.day,tzinfo=tz)
    dec=int((mid-dt.timedelta(hours=24)).timestamp())
    return city,d,dec
