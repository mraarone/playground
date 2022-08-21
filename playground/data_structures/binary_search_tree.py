import time

poem = """
"Hope" is the thing with feathers

"Hope" is the thing with feathers that perches in the soul of the bird-
And sings the tune without the words-
And never stops at all-


And sweetest in the gale is heard-
And sore must be the storm-
That kept so many warm-


I've heard it in the chillest land-
And on the strangest sea-
Yet never in extremity-
It asked a crumb of me.
"""

lines = poem.split('\n')

for line in lines:
    print(line)
    time.sleep(2)
