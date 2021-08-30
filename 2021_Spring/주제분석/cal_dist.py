# -*- coding: utf-8 -*-
from urllib.request import Request, urlopen
from urllib import parse
import json
import googlemaps


def with_bus(start, end, apikey):
    # need import googlemaps, pip install googlemaps
    #apikey = google cloud platform api key
    gmaps = googlemaps.Client(key=apikey)
    result = gmaps.directions(start, end, mode='transit')[0]['legs'][0]
    return result['distance'], result['duration']
   
def change_geocode(location, nid, npw):
    #nid npw = naver cloud platform 
    api_url = 'https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query='
    url = api_url+location
    request = Request(url)
    request.add_header('X-NCP-APIGW-API-KEY-ID', nid)
    request.add_header('X-NCP-APIGW-API-KEY', npw)
    response = urlopen(request)
    rescode = response.getcode() #정상이면 200
    if rescode == 200:
        res_body = response.read().decode('utf-8')
        res_body = json.loads(res_body)
        try:
            x = res_body['addresses'][0]['x']
            y = res_body['addresses'][0]['y']
            return x+','+y
        except:
            return 'NaN'
    else:
        return str(rescode),' error!'



def with_driving(start, end, nid, npw):
    driving_url = f'https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving?start={start}&goal={end}'
    request = Request(driving_url)
    request.add_header('X-NCP-APIGW-API-KEY-ID', nid)
    request.add_header('X-NCP-APIGW-API-KEY', npw)
    response = urlopen(request)
    rescode = response.getcode() #정상이면 200
    if rescode== 200:
        res_body = response.read().decode('utf-8')
        res_body = json.loads(res_body)
        distance = res_body['route']['traoptimal'][0]['summary']['distance'],'meters'
        duration = res_body['route']['traoptimal'][0]['summary']['duration'],'milisecond'
        return distance, duration
    else:
        return str(rescode),' error'
