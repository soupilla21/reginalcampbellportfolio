// Appends one event to data/feedback.json (spec §15), keeping each posting's events in
// strictly increasing time order so "latest wins" is never ambiguous.
import { fileErrors, type FeedbackEvent, type FeedbackFile } from "./feedback";
import { PATHS } from "./github";
import { appendWithRetry } from "./store";

export async function appendFeedbackEvent(event: FeedbackEvent): Promise<FeedbackEvent> {
  await appendWithRetry<FeedbackFile>({
    path: PATHS.feedback,
    empty: () => ({ schema_version: "1.0", updated_at: event.timestamp, events: [] }),
    validate: fileErrors,
    apply: (file) => {
      const prev = file.events.filter((e) => e.posting_id === event.posting_id).map((e) => e.timestamp).sort().pop();
      if (prev && Date.parse(prev) >= Date.parse(event.timestamp)) {
        event.timestamp = new Date(Date.parse(prev) + 1000).toISOString().replace(/\.\d{3}Z$/, "Z");
      }
      return { schema_version: "1.0", updated_at: event.timestamp, events: [...file.events, event] };
    },
    message: `feedback: ${event.posting_id} -> ${event.status}`,
  });
  return event;
}
