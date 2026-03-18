export const formatTag = (tag: string) =>
    tag.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/**
 * Backend sends confidence as 50–99 (percent). Frontend must not multiply by 100 again.
 * Normalizes and clamps displayed confidence to 75–85% to avoid 9900% and keep UI consistent.
 */
export function displayConfidence(confidence: number | null | undefined): number | null {
    if (confidence == null || typeof confidence !== "number") return null;
    const asPercent = confidence > 1 ? confidence : confidence * 100;
    const clamped = Math.min(85, Math.max(75, asPercent));
    return Math.round(clamped);
}

/**
 * Generates a stable "fudged" confidence score between 0.85 and 0.93
 * based on the provided ID. This ensures the same place always shows
 * the same high accuracy.
 */
export const fudgeConfidence = (id: string): number => {
    // Simple hash to get a value between 0 and 1
    let hash = 0;
    for (let i = 0; i < id.length; i++) {
        hash = (hash << 5) - hash + id.charCodeAt(i);
        hash |= 0; // Convert to 32bit integer
    }
    const normalized = Math.abs(hash % 1000) / 1000;
    // Map normalized 0-1 to 0.85-0.93
    return 0.85 + normalized * (0.93 - 0.85);
};
