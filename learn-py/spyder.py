from urllib.request import urlopen

html = urlopen(
    "https://mars.nasa.gov/msl/multimedia/raw-images/?order=sol+desc%2Cinstrument_sort+asc%2Csample_type_sort+asc%2C+date_taken+desc&per_page=50&page=0&mission=msl"
).read().decode('utf-8')

print(html)