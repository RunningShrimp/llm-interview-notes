/**
 * Table of Contents (TOC) Handler
 * Manages interactivity for TOC including smooth scrolling, active link highlighting,
 * and scroll position tracking.
 */

class TableOfContents {
  constructor(options = {}) {
    this.tocContainer = document.querySelector(options.container || '.toc');
    this.contentArea = document.querySelector(options.contentArea || 'main');
    this.activeClass = options.activeClass || 'active';
    this.highlightOffset = options.highlightOffset || 100;
    this.smoothScroll = options.smoothScroll !== false;
    this.trackScrollPosition = options.trackScrollPosition !== false;
    this.storageKey = options.storageKey || 'toc-scroll-position';
    
    // Cache for heading elements
    this.headings = [];
    this.tocLinks = [];
    
    // Debounce timer for scroll events
    this.scrollTimeout = null;
    this.scrollThrottle = options.scrollThrottle || 100;
    
    if (this.tocContainer) {
      this.init();
    }
  }

  /**
   * Initialize the TOC handler
   */
  init() {
    this.cacheHeadings();
    this.cacheTocLinks();
    this.attachEventListeners();
    this.restoreScrollPosition();
    this.updateActiveLink();
  }

  /**
   * Cache all headings in the content area
   */
  cacheHeadings() {
    if (!this.contentArea) return;
    
    // Target common heading selectors (h1, h2, h3, etc.)
    this.headings = Array.from(
      this.contentArea.querySelectorAll('h1, h2, h3, h4, h5, h6')
    ).filter(heading => {
      // Filter out headings without IDs or text
      return heading.id && heading.textContent.trim();
    });
  }

  /**
   * Cache all TOC links
   */
  cacheTocLinks() {
    if (!this.tocContainer) return;
    
    this.tocLinks = Array.from(this.tocContainer.querySelectorAll('a[href^="#"]'));
  }

  /**
   * Attach event listeners
   */
  attachEventListeners() {
    // Handle TOC link clicks for smooth scrolling
    this.tocLinks.forEach(link => {
      link.addEventListener('click', (e) => this.handleLinkClick(e));
    });

    // Handle scroll events for active link highlighting
    window.addEventListener('scroll', () => this.handleScroll());

    // Save scroll position before unload
    window.addEventListener('beforeunload', () => this.saveScrollPosition());

    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
      if (document.visible) {
        this.updateActiveLink();
      }
    });
  }

  /**
   * Handle TOC link click
   */
  handleLinkClick(e) {
    const href = e.currentTarget.getAttribute('href');
    const targetId = href.substring(1);
    const targetElement = document.getElementById(targetId);

    if (targetElement) {
      e.preventDefault();

      if (this.smoothScroll) {
        this.smoothScrollTo(targetElement);
      } else {
        targetElement.scrollIntoView();
      }

      // Update active link immediately
      this.setActiveLink(e.currentTarget);
    }
  }

  /**
   * Smooth scroll to element
   */
  smoothScrollTo(element) {
    const targetPosition = element.getBoundingClientRect().top + window.scrollY - this.highlightOffset;
    
    window.scrollTo({
      top: targetPosition,
      behavior: 'smooth'
    });
  }

  /**
   * Handle scroll event with throttling
   */
  handleScroll() {
    clearTimeout(this.scrollTimeout);
    
    this.scrollTimeout = setTimeout(() => {
      this.updateActiveLink();
      
      if (this.trackScrollPosition) {
        this.saveScrollPosition();
      }
    }, this.scrollThrottle);
  }

  /**
   * Update active link based on current scroll position
   */
  updateActiveLink() {
    const scrollPosition = window.scrollY + this.highlightOffset;
    let activeHeading = null;

    // Find the heading that is currently in view
    for (const heading of this.headings) {
      const headingPosition = heading.getBoundingClientRect().top + window.scrollY;
      
      if (headingPosition <= scrollPosition) {
        activeHeading = heading;
      } else {
        break;
      }
    }

    // Update active link
    if (activeHeading) {
      const correspondingLink = this.tocLinks.find(
        link => link.getAttribute('href') === `#${activeHeading.id}`
      );
      
      if (correspondingLink) {
        this.setActiveLink(correspondingLink);
      }
    }
  }

  /**
   * Set active link and update DOM
   */
  setActiveLink(linkElement) {
    // Remove active class from all links
    this.tocLinks.forEach(link => {
      link.classList.remove(this.activeClass);
      
      // Also remove from parent list items if they exist
      const listItem = link.closest('li');
      if (listItem) {
        listItem.classList.remove(this.activeClass);
      }
    });

    // Add active class to current link
    linkElement.classList.add(this.activeClass);
    
    // Add active class to parent list item
    const listItem = linkElement.closest('li');
    if (listItem) {
      listItem.classList.add(this.activeClass);
    }

    // Scroll TOC container if needed to keep active link visible
    this.scrollTocIntoView(linkElement);
  }

  /**
   * Scroll TOC container to show active link
   */
  scrollTocIntoView(linkElement) {
    if (!this.tocContainer) return;

    const tocRect = this.tocContainer.getBoundingClientRect();
    const linkRect = linkElement.getBoundingClientRect();

    // Check if link is outside TOC container bounds
    if (linkRect.top < tocRect.top || linkRect.bottom > tocRect.bottom) {
      linkElement.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
      });
    }
  }

  /**
   * Save current scroll position to localStorage
   */
  saveScrollPosition() {
    if (!this.trackScrollPosition) return;

    try {
      const scrollData = {
        timestamp: Date.now(),
        position: window.scrollY,
        url: window.location.href
      };
      
      localStorage.setItem(this.storageKey, JSON.stringify(scrollData));
    } catch (e) {
      console.warn('Failed to save scroll position:', e);
    }
  }

  /**
   * Restore previous scroll position
   */
  restoreScrollPosition() {
    if (!this.trackScrollPosition) return;

    try {
      const scrollData = JSON.parse(localStorage.getItem(this.storageKey));
      
      if (scrollData && scrollData.url === window.location.href) {
        // Use setTimeout to ensure page is fully rendered
        setTimeout(() => {
          window.scrollTo(0, scrollData.position);
          this.updateActiveLink();
        }, 100);
      }
    } catch (e) {
      console.warn('Failed to restore scroll position:', e);
    }
  }

  /**
   * Generate TOC from headings dynamically
   */
  static generateToc(contentSelector, options = {}) {
    const contentArea = document.querySelector(contentSelector);
    if (!contentArea) return null;

    const headings = Array.from(contentArea.querySelectorAll('h1, h2, h3, h4, h5, h6'));
    let lastLevel = 0;
    let html = '';
    let stack = [];

    headings.forEach((heading, index) => {
      // Skip headings without IDs
      if (!heading.id) {
        heading.id = `heading-${index}`;
      }

      const level = parseInt(heading.tagName[1]);
      const text = heading.textContent.trim();
      const href = `#${heading.id}`;

      // Adjust nesting level
      while (lastLevel < level) {
        html += '<ul>';
        stack.push('</ul>');
        lastLevel++;
      }

      while (lastLevel > level) {
        html += stack.pop();
        lastLevel--;
      }

      html += `<li><a href="${href}">${text}</a></li>`;
    });

    // Close remaining open lists
    while (stack.length > 0) {
      html += stack.pop();
    }

    const wrapper = document.createElement('nav');
    wrapper.className = options.className || 'toc';
    wrapper.innerHTML = html;

    return wrapper;
  }

  /**
   * Destroy the TOC handler and remove event listeners
   */
  destroy() {
    if (this.scrollTimeout) {
      clearTimeout(this.scrollTimeout);
    }

    window.removeEventListener('scroll', () => this.handleScroll());
    window.removeEventListener('beforeunload', () => this.saveScrollPosition());
    document.removeEventListener('visibilitychange', () => this.updateActiveLink());

    this.tocContainer = null;
    this.contentArea = null;
    this.headings = [];
    this.tocLinks = [];
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TableOfContents;
}
