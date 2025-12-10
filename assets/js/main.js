// GitHub Pages Interactive Features

(function() {
  'use strict';

  // ========================================
  // 1. Table of Contents Generation
  // ========================================
  function generateTOC() {
    const content = document.querySelector('.main-content');
    const sidebar = document.querySelector('.sidebar');
    
    if (!content || !sidebar) return;
    
    const headings = content.querySelectorAll('h2, h3');
    if (headings.length === 0) return;
    
    const tocList = document.createElement('ul');
    tocList.className = 'toc-list';
    
    headings.forEach((heading, index) => {
      // Add ID to heading for anchor links
      if (!heading.id) {
        heading.id = 'heading-' + index;
      }
      
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = '#' + heading.id;
      a.textContent = heading.textContent;
      a.className = heading.tagName === 'H3' ? 'toc-h3' : 'toc-h2';
      
      li.appendChild(a);
      tocList.appendChild(li);
    });
    
    const tocTitle = document.createElement('h3');
    tocTitle.textContent = '目录导航';
    
    sidebar.innerHTML = '';
    sidebar.appendChild(tocTitle);
    sidebar.appendChild(tocList);
  }

  // ========================================
  // 2. Active TOC Item Highlighting
  // ========================================
  function updateActiveTOC() {
    const headings = document.querySelectorAll('.main-content h2, .main-content h3');
    const tocLinks = document.querySelectorAll('.toc-list a');
    
    if (headings.length === 0 || tocLinks.length === 0) return;
    
    let activeIndex = -1;
    const scrollPosition = window.scrollY + 100;
    
    headings.forEach((heading, index) => {
      if (heading.offsetTop <= scrollPosition) {
        activeIndex = index;
      }
    });
    
    tocLinks.forEach((link, index) => {
      if (index === activeIndex) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });
  }

  // ========================================
  // 3. Back to Top Button
  // ========================================
  function initBackToTop() {
    const button = document.querySelector('.back-to-top');
    if (!button) return;
    
    function toggleButton() {
      if (window.scrollY > 300) {
        button.classList.add('visible');
      } else {
        button.classList.remove('visible');
      }
    }
    
    window.addEventListener('scroll', toggleButton);
    toggleButton();
    
    button.addEventListener('click', function() {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  }

  // ========================================
  // 4. Theme Toggle
  // ========================================
  function initThemeToggle() {
    const toggle = document.querySelector('.theme-toggle');
    if (!toggle) return;
    
    // Check for saved theme preference or default to system preference
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
      document.body.classList.add('dark-mode');
      toggle.textContent = '☀️';
    } else {
      toggle.textContent = '🌙';
    }
    
    toggle.addEventListener('click', function() {
      document.body.classList.toggle('dark-mode');
      
      if (document.body.classList.contains('dark-mode')) {
        localStorage.setItem('theme', 'dark');
        toggle.textContent = '☀️';
      } else {
        localStorage.setItem('theme', 'light');
        toggle.textContent = '🌙';
      }
    });
  }

  // ========================================
  // 5. Copy Code Button
  // ========================================
  function initCopyCodeButtons() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(function(codeBlock) {
      const pre = codeBlock.parentElement;
      
      // Skip if already wrapped
      if (pre.parentElement.classList.contains('code-wrapper')) return;
      
      // Wrap pre in a div
      const wrapper = document.createElement('div');
      wrapper.className = 'code-wrapper';
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);
      
      // Create copy button
      const button = document.createElement('button');
      button.className = 'copy-button';
      button.textContent = '复制';
      button.setAttribute('aria-label', '复制代码');
      
      button.addEventListener('click', function() {
        const code = codeBlock.textContent;
        
        navigator.clipboard.writeText(code).then(function() {
          button.textContent = '已复制!';
          button.classList.add('copied');
          
          setTimeout(function() {
            button.textContent = '复制';
            button.classList.remove('copied');
          }, 2000);
        }).catch(function(err) {
          console.error('复制失败:', err);
          button.textContent = '失败';
          setTimeout(function() {
            button.textContent = '复制';
          }, 2000);
        });
      });
      
      wrapper.appendChild(button);
    });
  }

  // ========================================
  // 6. Smooth Scrolling for Anchor Links
  // ========================================
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
      anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href === '#') return;
        
        const target = document.querySelector(href);
        if (target) {
          e.preventDefault();
          const offsetTop = target.offsetTop - 80;
          window.scrollTo({
            top: offsetTop,
            behavior: 'smooth'
          });
        }
      });
    });
  }

  // ========================================
  // 7. Keyboard Navigation
  // ========================================
  function initKeyboardNav() {
    document.addEventListener('keydown', function(e) {
      // Press 't' to toggle theme
      if (e.key === 't' && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
        const activeElement = document.activeElement;
        if (activeElement.tagName !== 'INPUT' && activeElement.tagName !== 'TEXTAREA') {
          const toggle = document.querySelector('.theme-toggle');
          if (toggle) toggle.click();
        }
      }
      
      // Press ESC to go back to top
      if (e.key === 'Escape') {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  }

  // ========================================
  // 8. Enhanced Tables
  // ========================================
  function enhanceTables() {
    const tables = document.querySelectorAll('table');
    tables.forEach(function(table) {
      // Wrap table in a responsive container if not already wrapped
      if (!table.parentElement.classList.contains('table-wrapper')) {
        const wrapper = document.createElement('div');
        wrapper.className = 'table-wrapper';
        wrapper.style.overflowX = 'auto';
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
      }
    });
  }

  // ========================================
  // 9. External Links
  // ========================================
  function markExternalLinks() {
    const links = document.querySelectorAll('a[href^="http"]');
    links.forEach(function(link) {
      if (!link.hostname.includes(window.location.hostname)) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
      }
    });
  }

  // ========================================
  // 10. Print Preparation
  // ========================================
  function preparePrint() {
    window.addEventListener('beforeprint', function() {
      // Expand all collapsed sections if any
      document.querySelectorAll('details').forEach(function(details) {
        details.setAttribute('open', '');
      });
    });
  }

  // ========================================
  // Initialize All Features
  // ========================================
  function init() {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }
    
    console.log('Initializing GitHub Pages enhancements...');
    
    generateTOC();
    initBackToTop();
    initThemeToggle();
    initCopyCodeButtons();
    initSmoothScroll();
    initKeyboardNav();
    enhanceTables();
    markExternalLinks();
    preparePrint();
    
    // Update active TOC on scroll
    let scrollTimeout;
    window.addEventListener('scroll', function() {
      if (scrollTimeout) {
        window.cancelAnimationFrame(scrollTimeout);
      }
      scrollTimeout = window.requestAnimationFrame(function() {
        updateActiveTOC();
      });
    });
    
    // Initial TOC update
    updateActiveTOC();
    
    console.log('GitHub Pages enhancements initialized successfully!');
  }

  // Start initialization
  init();
})();
