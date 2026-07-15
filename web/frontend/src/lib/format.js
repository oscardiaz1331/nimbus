// Cloud metrics arrive as [0,1] fractions (system-wide convention);
// formatting to % happens here, at the presentation edge.

export const pct = (fraction) => `${Math.round(fraction * 100)}%`;

export const timeShort = (date) =>
  new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

export const dateTime = (date) => new Date(date).toLocaleString();
