import requests
import json

player_id = 8471675

url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"

response = requests.get(url)

# Check success
print("Status:", response.status_code)

data = response.json()

# Save everything
with open("crosby_full_nhle.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved to crosby_full_nhle.json")
