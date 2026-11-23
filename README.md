# refind

Find a web element again after the page has changed.

When a browser agent finishes a task, the cheapest way to repeat it is to replay the recorded actions without calling the model. The part that breaks is locating elements. Recorded XPaths and CSS selectors stop working as soon as a banner is added, a wrapper div appears, or the build regenerates class names. `refind` records a fingerprint of each element instead and uses it to find the same element in the new DOM. If it is not sure, it refuses to answer, so the replay can fall back to the model for that one step.

It is one file, about 230 lines of Python, with no dependencies. Copy `refind.py` into a project to use it.

## Results

`python bench.py` takes every interactive element on three fixture pages (a login form, a product grid, an admin table: 63 elements), records three locators for it, applies 1 to 3 random page mutations, and checks which locators still resolve to the same element. 40 trials per element.

| mutations | trials | absolute XPath | CSS selector | refind | refind wrong element | refind refused |
|---|---|---|---|---|---|---|
| 1 | 2520 | 55.9% | 12.5% | 87.3% | 0.0% | 12.7% |
| 2 | 2520 | 30.8% | 10.1% | 86.9% | 0.0% | 13.1% |
| 3 | 2520 | 14.3% | 8.4% | 85.9% | 0.0% | 14.1% |

The mutations are: insert a cookie banner, wrap the element's parent in a div, regenerate every class name, regenerate every id, shuffle the surrounding siblings, insert a same-tag sibling before the element, and change the element's text.

The column I care about most is "wrong element". A replay that clicks the wrong button is worse than one that stops. Every refusal in the table is a case where two candidates scored within 0.04 of each other, mostly the repeated Edit and Remove buttons in the admin table.

## How it works

`fingerprint(node)` stores the tag, attributes that describe what the element is (`name`, `aria-label`, `data-testid`, `role`, `type`, `placeholder`, `href`), its visible text, the tags of its ancestors, and the text of its siblings. Ids and class names that look generated (`css-1x2y3z`, `sc-a1b2c3`, long hex strings) are dropped at record time.

`find(fingerprint, root)` scores every element with the same tag. Each signal contributes only when there is evidence for it on at least one side, so an element with no text is not penalised for having no text. It returns a `Match` with a confidence and the margin over the runner-up, and returns no node when the best score is under 0.55 or the margin is under 0.04.

```python
from refind import parse, by_xpath, fingerprint, find

old = parse(open("checkout_monday.html").read())
fp = fingerprint(by_xpath(old, "/html[1]/body[1]/main[1]/form[1]/button[1]"))

new = parse(open("checkout_friday.html").read())
m = find(fp, new)
if m.node is None:
    ...  # hand this step back to the LLM
else:
    print(m.method, m.confidence, m.node.xpath())
```

There is also a CLI: `python refind.py record page.html <xpath>` and `python refind.py find fp.json new_page.html`.

## What I tried that did not work

I added a "row context" signal (the text of the enclosing table row or card) to separate the repeated Edit buttons. It cut some refusals but produced wrong matches for the first time (0.6% to 1.2%), because shuffling rows changes the context of every button at once. I took it out. The right fix is probably to fingerprint the row and the button as a pair.

## Limits

- The benchmark pages and mutations are synthetic. I have not run this on recorded sessions from real sites yet.
- It works on static HTML. It is not wired into browser-use or Playwright. The intended hook is the action history: store a fingerprint next to each recorded element index, and on rerun call `find` on the current DOM snapshot before asking the model.
- It does not look inside iframes or shadow roots.

## Run it

```
python test_refind.py
python bench.py
```

Python 3.9 or newer. Nothing to install.
