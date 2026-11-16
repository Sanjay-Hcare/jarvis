// Jarvis calendar bridge (read-only).
//
// 1. Open https://script.google.com, create a new project and paste this file in.
// 2. Replace the SECRET below with the key printed by: python -m jarvis.setup_accounts
// 3. Select the "authorize" function and click Run once to grant calendar access.
// 4. Deploy > New deployment > Web app. Execute as: Me. Who has access: Anyone.
// 5. Copy the web app URL and paste it into the setup command.
//
// Anyone who has both the URL and the SECRET can read your calendar, so keep
// them private. The script can only list events; it cannot change anything.
const SECRET = 'PASTE_THE_KEY_FROM_THE_SETUP_COMMAND_HERE';

function doPost(e) {
  let d;
  try { d = JSON.parse(e.postData.contents); } catch (err) { return out({ok: false, error: 'bad request'}); }
  if (d.secret !== SECRET) return out({ok: false, error: 'unauthorized'});
  const cal = CalendarApp.getDefaultCalendar();
  const tz = Session.getScriptTimeZone();
  try {
    if (d.action === 'list') {
      const evs = cal.getEvents(new Date(d.start), new Date(d.end)).map(x => ({
        title: x.getTitle(),
        all_day: x.isAllDayEvent(),
        date: Utilities.formatDate(x.getStartTime(), tz, 'yyyy-MM-dd'),
        start: x.getStartTime().toISOString(),
        end: x.getEndTime().toISOString(),
      }));
      return out({ok: true, events: evs});
    }
    return out({ok: false, error: 'unknown action'});
  } catch (err) {
    return out({ok: false, error: String(err)});
  }
}

function out(o) { return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON); }

// Run this once from the editor to grant calendar access.
function authorize() { CalendarApp.getDefaultCalendar().getName(); }
