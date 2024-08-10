---
layout: post
icon: fas fa-camera
order: 1
toc: true
---

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
