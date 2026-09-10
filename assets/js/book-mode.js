document.addEventListener('DOMContentLoaded', function () {
  var toggle = document.getElementById('book-mode-toggle');
  var content = document.querySelector('article .content');
  if (!toggle || !content) return;

  /* Chirpy's sidebar/mobile TOC scroll-spy assumes every heading is laid out
     in normal document flow. Once book mode hides all but the current
     section, headings in hidden sections report no layout box, which makes
     the scroll-spy highlight the wrong entry and makes TOC links unable to
     scroll to their target. Simplest reliable fix: TOC and book mode are
     mutually exclusive — hide the TOC entirely while paginated. */
  var tocEls = ['toc-wrapper', 'toc-bar', 'toc-solo-trigger', 'toc-popup']
    .map(function (id) { return document.getElementById(id); })
    .filter(Boolean);

  var tocWrapper = document.querySelector('.book-toc-wrapper');
  var tocDetails = tocWrapper ? tocWrapper.querySelector('.book-toc') : null;
  var tocList = tocWrapper ? tocWrapper.querySelector('.book-toc-list') : null;

  var sections = buildSections();
  var currentIndex = 0;
  var active = false;

  if (sections && sections.length > 1) buildTocList();

  if (tocDetails) {
    document.addEventListener('click', function (e) {
      if (tocDetails.open && !tocDetails.contains(e.target)) tocDetails.open = false;
    });
  }

  function sectionTitle(group, index) {
    var heading = group[0].tagName.match(/^H[1-6]$/) ? group[0] : null;
    if (heading) return heading.textContent.trim();
    return index === 0 ? 'Introduction' : 'Section ' + (index + 1);
  }

  function buildTocList() {
    if (!tocList) return;
    sections.forEach(function (group, i) {
      var link = document.createElement('a');
      link.href = '#';
      link.className = 'book-toc-link';
      link.textContent = sectionTitle(group, i);
      link.addEventListener('click', function (e) {
        e.preventDefault();
        showSection(i);
        if (tocDetails) tocDetails.open = false;
      });
      tocList.appendChild(link);
    });
  }

  function updateTocHighlight() {
    if (!tocList) return;
    Array.prototype.forEach.call(tocList.children, function (link, i) {
      link.classList.toggle('active', i === currentIndex);
    });
  }

  function buildSections() {
    var minLevel = null;
    Array.prototype.forEach.call(content.children, function (node) {
      var match = node.tagName.match(/^H([1-6])$/);
      if (match) {
        var level = parseInt(match[1], 10);
        if (minLevel === null || level < minLevel) minLevel = level;
      }
    });
    if (minLevel === null) return null;

    var groups = [];
    var current = [];
    Array.prototype.forEach.call(content.children, function (node) {
      var match = node.tagName.match(/^H([1-6])$/);
      var level = match ? parseInt(match[1], 10) : null;
      if (level === minLevel && current.length) {
        groups.push(current);
        current = [];
      }
      current.push(node);
    });
    if (current.length) groups.push(current);
    return groups;
  }

  function headingGroupIndex(heading) {
    for (var i = 0; i < sections.length; i++) {
      if (sections[i].indexOf(heading) !== -1) return i;
    }
    return -1;
  }

  function setGroupVisible(group, visible) {
    group.forEach(function (node) {
      node.style.display = visible ? '' : 'none';
    });
  }

  function updateNav() {
    var label = 'Page ' + (currentIndex + 1) + ' of ' + sections.length;
    document.querySelectorAll('.book-page-indicator').forEach(function (el) {
      el.textContent = label;
    });
    document.querySelectorAll('.book-prev').forEach(function (btn) {
      btn.disabled = currentIndex === 0;
    });
    document.querySelectorAll('.book-next').forEach(function (btn) {
      btn.disabled = currentIndex === sections.length - 1;
    });
  }

  function showSection(index, scroll) {
    sections.forEach(function (group, i) {
      setGroupVisible(group, i === index);
    });
    currentIndex = index;
    updateNav();
    updateTocHighlight();
    /* nudge any widget that sizes itself off window resize (charts, canvas,
       ResizeObserver-driven components) now that its container just went
       from zero to real height */
    window.dispatchEvent(new Event('resize'));
    if (scroll !== false) {
      var top = content.getBoundingClientRect().top + window.scrollY - 80;
      window.scrollTo(0, top);
    }
  }

  function setTocVisible(visible) {
    tocEls.forEach(function (el) {
      el.classList.toggle('d-none', !visible);
    });
  }

  function enableBookMode() {
    if (!sections || sections.length < 2) return;
    active = true;
    document.querySelectorAll('.book-nav').forEach(function (el) {
      el.classList.remove('d-none');
      el.classList.add('d-flex');
    });
    if (tocWrapper) tocWrapper.classList.remove('d-none');
    toggle.classList.add('active');
    setTocVisible(false);
    showSection(0, false);
  }

  function disableBookMode() {
    active = false;
    sections.forEach(function (group) { setGroupVisible(group, true); });
    document.querySelectorAll('.book-nav').forEach(function (el) {
      el.classList.add('d-none');
      el.classList.remove('d-flex');
    });
    if (tocWrapper) tocWrapper.classList.add('d-none');
    if (tocDetails) tocDetails.open = false;
    toggle.classList.remove('active');
    setTocVisible(true);
    window.dispatchEvent(new Event('resize'));
  }

  toggle.addEventListener('click', function () {
    if (active) {
      disableBookMode();
    } else {
      enableBookMode();
    }
  });

  document.querySelectorAll('.book-prev').forEach(function (btn) {
    btn.addEventListener('click', function () {
      if (currentIndex > 0) showSection(currentIndex - 1);
    });
  });

  document.querySelectorAll('.book-next').forEach(function (btn) {
    btn.addEventListener('click', function () {
      if (currentIndex < sections.length - 1) showSection(currentIndex + 1);
    });
  });

  /* In-page anchor links (footnotes, "back to top", any leftover TOC link)
     that target a heading living in a section other than the current one
     need to switch pages first — a plain browser jump can't scroll to an
     element with no layout box. */
  document.addEventListener('click', function (e) {
    if (!active || !sections) return;
    var link = e.target.closest('a[href^="#"]');
    if (!link) return;
    var id = link.getAttribute('href').slice(1);
    if (!id) return;
    var target = document.getElementById(id);
    if (!target || !content.contains(target)) return;
    var heading = target.tagName.match(/^H[1-6]$/) ? target : target.closest('h1, h2, h3, h4, h5, h6');
    if (!heading) return;
    var groupIndex = headingGroupIndex(heading);
    if (groupIndex === -1 || groupIndex === currentIndex) return;
    e.preventDefault();
    showSection(groupIndex, false);
    requestAnimationFrame(function () {
      target.scrollIntoView({ block: 'start' });
    });
  });
});
