---
layout: post
icon: fas fa-camera
order: 1
toc: true
---

## Welcome!

![Image]({{ site.baseurl }}/assets/images/photography/main/young.png){: width="300"}

This is my photography page.  As you can see from the photo above, I have been loving taking photos since a long time!  I love traveling to different places and trying to capture what I had experienced and saw to share with others.  Most photos are unedited, and I really didn't wait for certain conditions - rather just tried to get the best shot with what I had at the time.

**Please click on the photos** and use the arrow keys to **view the photos in fullscreen.**  Also, please check out the table of contents on the right side to skip through sections.

![Image]({{ site.baseurl }}/assets/images/photography/main/today.png){: width="500"}

## Norway

{% for image in site.static_files %}
    {% if image.path contains 'assets/images/photography/norway' %}
<img src="{{ site.baseurl }}{{ image.path }}" alt="image">
    {% endif %}
{% endfor %}

---

## England, Scotland, and Ireland

{% for image in site.static_files %}
    {% if image.path contains 'assets/images/photography/englandETC' %}
<img src="{{ site.baseurl }}{{ image.path }}" alt="image">
    {% endif %}
{% endfor %}

---

## Lunar Eclipse

![Image]({{ site.baseurl }}/assets/images/photography/lunarEclipse.jpg)
<figcaption>Taken on Nov. 8, 2022 in Maryland</figcaption>


---

## California Bay Area

{% for image in site.static_files %}
    {% if image.path contains 'assets/images/photography/california' %}
<img src="{{ site.baseurl }}{{ image.path }}" alt="image">
    {% endif %}
{% endfor %}

---

## State of Washington

{% for image in site.static_files %}
    {% if image.path contains 'assets/images/photography/seattle' %}
<img src="{{ site.baseurl }}{{ image.path }}" alt="image">
    {% endif %}
{% endfor %}

---


## Belize

{% for image in site.static_files %}
    {% if image.path contains 'assets/images/photography/belize' %}
<img src="{{ site.baseurl }}{{ image.path }}" alt="image">
    {% endif %}
{% endfor %}
