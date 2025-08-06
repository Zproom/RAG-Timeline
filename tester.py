x = """
{
    "events": [
        {
            "start_date": { "year": 2023 },
            "text": {
                "headline": "Trump Zeroes in on Undocumented Immigration",
                "text": "Trump has zeroed in on undocumented immigration in his second term after promising to do just that throughout the campaign."
            }
        },
        {
            "start_date": { "year": 2023 },
            "text": {
                "headline": "Trump Launches Military Mission at Southern Border",
                "text": "Since taking office, he launched a costly military mission at the Southern border."
            }
        },
        {
            "start_date": { "year": 2023 },
            "text": {
                "headline": "Trump Deports Undocumented Migrants with Alleged Gang Ties",
                "text": "Trump deported undocumented migrants with alleged gang ties to a mega prison in El Salvador."
            }
        },
        {
            "start_date": { "year": 2023 },
            "text": {
                "headline": "Trump Floats Sending More People to Libya, Rwanda and Saudi Arabia",
                "text": "Immigrants make up a larger percentage of the population in states along the East and West coasts and the Southern border. Trump has said illegal immigration to the United States amounts to an �invasion,� using the term in his executive orders and agency memos."
            }
        },
        {
            "start_date": { "year": 2023 },
            "text": {
                "headline": "California, New Jersey and New York Challenge Trump�s Immigration Moves",
                "text": "California, New Jersey and New York � all Democrat-led states where immigrants make up the highest share of the populations � have challenged Trump�s immigration moves, including filing a lawsuit against his birthright citizenship executive order and against the administration�s requirements tying federal grant funding to state participation in ongoing immigration enforcement efforts."
            }
        }
    ]
}
"""

import re
import json

event_pattern = re.compile(r'\{[^{}]*"start_date"[^{}]*\{[^{}]*"year"[^{}]*\}[^{}]*"text"[^{}]*\{[^{}]*"headline"[^{}]*"text"[^{}]*\}[^{}]*\}', re.DOTALL)
event_matches = event_pattern.findall(x)
clean_events = []
for event_str in event_matches:
    try:
        json_str = event_str.replace("'", '"')
        event = json.loads(json_str)

        clean_events.append(event)

    except:
        continue

json_string = json.loads(str({"events": clean_events}))

with open("test.json", "w+") as f:
    json.dump(json_string, f, indent=4)