/* Shared, non-modal mobile navigation. Desktop navigation always stays visible. */
const sidebar = document.querySelector('.sidebar');
const nav = sidebar?.querySelector('nav');
const brand = sidebar?.querySelector('.brand');

if (nav && brand) {
  const mobile = window.matchMedia('(max-width: 900px)');
  nav.id ||= 'viewer-navigation';
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'mobile-nav-toggle';
  toggle.setAttribute('aria-controls', nav.id);
  toggle.setAttribute('aria-expanded', 'false');
  toggle.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16"/></svg><span>Menu</span>';
  brand.append(toggle);
  sidebar.classList.add('mobile-nav-ready');

  const close = ({ restoreFocus = false } = {}) => {
    sidebar.classList.remove('mobile-nav-open');
    toggle.setAttribute('aria-expanded', 'false');
    nav.inert = mobile.matches;
    if (restoreFocus) toggle.focus();
  };
  const open = () => {
    sidebar.classList.add('mobile-nav-open');
    toggle.setAttribute('aria-expanded', 'true');
    nav.inert = false;
  };
  toggle.addEventListener('click', () => {
    if (toggle.getAttribute('aria-expanded') === 'true') close();
    else open();
  });
  // Keep focus on the disclosure button on open; Tab reaches the first link.
  // This is an in-flow disclosure, so no modal focus trap is needed.
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && mobile.matches && sidebar.classList.contains('mobile-nav-open')) {
      event.preventDefault();
      close({ restoreFocus: true });
    }
  });
  document.addEventListener('click', event => {
    if (mobile.matches && !sidebar.contains(event.target)) close();
  });
  nav.addEventListener('click', event => {
    if (mobile.matches && event.target.closest('a, button')) close({ restoreFocus: true });
  });
  mobile.addEventListener('change', () => {
    const focusInNav = nav.contains(document.activeElement);
    close({ restoreFocus: mobile.matches && focusInNav });
  });
  close();
}

// Tag menus are native details, with the same dismiss shortcuts as navigation.
document.querySelectorAll('.tag-filter').forEach(details => {
  document.addEventListener('click', event => {
    if (!details.contains(event.target)) details.open = false;
  });
  details.addEventListener('keydown', event => {
    if (event.key === 'Escape' && details.open) {
      details.open = false;
      details.querySelector('summary').focus();
      event.preventDefault();
    }
  });
});
