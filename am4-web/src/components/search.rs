use leptos::html::Input;
use leptos::prelude::*;
use std::collections::HashMap;
use std::time::Duration;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::KeyboardEvent;

fn copy_to_clipboard(text: String) {
    if let Some(window) = web_sys::window() {
        let promise = window.navigator().clipboard().write_text(&text);
        spawn_local(async move {
            let _ = JsFuture::from(promise).await;
        });
    }
}

fn keyed_items<T: Clone>(
    items: Vec<T>,
    key: Callback<T, usize>,
) -> Vec<((usize, usize), usize, T)> {
    let mut occurrences = HashMap::<usize, usize>::with_capacity(items.len());
    items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            let base = key.run(item.clone());
            let occurrence = occurrences.entry(base).or_default();
            let stable_key = (base, *occurrence);
            *occurrence += 1;
            (stable_key, index, item)
        })
        .collect()
}

#[component]
pub fn MultiSelect<T, ViewOpt, ViewPill>(
    #[prop(into)] selected: RwSignal<Vec<T>>,
    #[prop(into)] active: RwSignal<Option<T>>,
    #[prop(into)] search: Callback<String, Vec<T>>,
    render_option: ViewOpt,
    render_pill: ViewPill,
    serialize: Callback<T, String>,
    key: Callback<T, usize>,
    #[prop(optional, into)] parse_token: Option<Callback<String, Option<T>>>,
    #[prop(optional, into)] max_items: Option<usize>,
    #[prop(optional, into)] placeholder: String,
    #[prop(optional, into)] is_selectable: Option<Callback<T, bool>>,
) -> impl IntoView
where
    T: Clone + PartialEq + Send + Sync + 'static,
    ViewOpt: Fn(
            T,
            usize,
            Memo<Vec<T>>,
            ReadSignal<usize>,
            Callback<T>,
            Callback<T>,
            Callback<Box<dyn Fn(T) -> T + Send + Sync>>,
        ) -> AnyView
        + 'static
        + Copy
        + Send
        + Sync,
    ViewPill: Fn(T, Callback<()>) -> AnyView + 'static + Copy + Send + Sync,
{
    let (query, set_query) = signal(String::new());
    let (is_focused, set_focused) = signal(false);
    let (highlight_idx, set_highlight_idx) = signal(0);
    let (pinned, set_pinned) = signal(None::<T>);
    let (cursor_idx, set_cursor_idx) = signal(None::<usize>);
    let (selection_anchor, set_selection_anchor) = signal(None::<usize>);
    let (selection_range, set_selection_range) = signal(None::<(usize, usize)>);
    let input_ref = NodeRef::<Input>::new();
    let suggestions_override = RwSignal::new(None::<Vec<T>>);

    let base_suggestions = Memo::new(move |_| {
        let q = query.get();
        if q.is_empty() {
            if pinned.get().is_none() {
                active.set(None);
            }
            suggestions_override.set(None);
            vec![]
        } else {
            suggestions_override.set(None);
            search.run(q)
        }
    });

    let suggestions = Memo::new(move |_| {
        suggestions_override
            .get()
            .unwrap_or_else(|| base_suggestions.get())
    });

    Effect::new(move |_| {
        suggestions.track();
        if let Some(cb) = is_selectable {
            let suggs = suggestions.get_untracked();
            if let Some(idx) = suggs.iter().position(|item| cb.run(item.clone())) {
                set_highlight_idx.set(idx);
            } else {
                set_highlight_idx.set(0);
            }
        } else {
            set_highlight_idx.set(0);
        }
    });

    let select = move |item: T| {
        if let Some(cb) = is_selectable {
            if !cb.run(item.clone()) {
                return;
            }
        }

        selected.update(|s| {
            if let Some(max) = max_items {
                if s.len() >= max {
                    s.pop();
                }
            }
            if !s.contains(&item) {
                s.push(item.clone());
            }
        });
        set_query.set(String::new());
        set_pinned.set(None);
        set_cursor_idx.set(None);
        set_selection_anchor.set(None);
        set_selection_range.set(None);
        active.set(None);
        suggestions_override.set(None);
        if let Some(input) = input_ref.get() {
            input.focus().ok();
        }
    };

    let remove = move |item: T| {
        selected.update(|s| {
            if let Some(pos) = s.iter().position(|x| *x == item) {
                s.remove(pos);
            }
        });
        if pinned.get() == Some(item) {
            set_pinned.set(None);
            active.set(None);
        }

        set_cursor_idx.update(|idx| {
            if let Some(i) = *idx {
                let len = selected.get_untracked().len();
                if len == 0 {
                    *idx = None;
                } else if i >= len {
                    *idx = Some(len - 1);
                }
            }
        });
    };

    let parse_and_apply_csv = move |value: String| {
        let Some(parse_token) = parse_token else {
            set_query.set(value);
            return;
        };

        let has_csv = value.contains(',');
        if !has_csv {
            set_query.set(value);
            return;
        }

        let parts: Vec<&str> = value.split(',').collect();
        let trailing_comma = value.ends_with(',');
        let committed_len = if trailing_comma {
            parts.len()
        } else {
            parts.len().saturating_sub(1)
        };

        let mut leftovers = Vec::<String>::new();

        for token in parts.iter().take(committed_len) {
            let trimmed = token.trim();
            if trimmed.is_empty() {
                continue;
            }
            match parse_token.run(trimmed.to_string()) {
                Some(item)
                    if is_selectable.is_none_or(|cb| cb.run(item.clone()))
                        && !selected.get_untracked().contains(&item) =>
                {
                    selected.update(|s| {
                        if let Some(max) = max_items {
                            if s.len() >= max {
                                s.pop();
                            }
                        }
                        if !s.contains(&item) {
                            s.push(item);
                        }
                    });
                }
                _ => leftovers.push(trimmed.to_string()),
            }
        }

        let tail = if trailing_comma {
            ""
        } else {
            parts.last().copied().unwrap_or_default().trim()
        };

        if !tail.is_empty() {
            leftovers.push(tail.to_string());
        }

        set_query.set(leftovers.join(","));
        set_cursor_idx.set(None);
        set_selection_anchor.set(None);
        set_selection_range.set(None);
    };

    let on_keydown = move |ev: KeyboardEvent| {
        let key = ev.key();
        let ctrl_or_meta = ev.ctrl_key() || ev.meta_key();
        let selected_items = selected.get();

        if ctrl_or_meta && key.eq_ignore_ascii_case("a") && query.get().is_empty() {
            if !selected_items.is_empty() {
                ev.prevent_default();
                set_selection_anchor.set(Some(0));
                set_selection_range.set(Some((0, selected_items.len() - 1)));
                set_cursor_idx.set(Some(selected_items.len() - 1));
                active.set(selected_items.last().cloned());
            }
            return;
        }

        if ctrl_or_meta && key.eq_ignore_ascii_case("c") {
            let mut copied = Vec::<String>::new();
            if let Some((start, end)) = selection_range.get() {
                copied = selected_items[start..=end]
                    .iter()
                    .cloned()
                    .map(|item| serialize.run(item))
                    .collect();
            } else if !selected_items.is_empty() && query.get().is_empty() {
                copied = selected_items
                    .into_iter()
                    .map(|item| serialize.run(item))
                    .collect();
            }

            if !copied.is_empty() {
                ev.prevent_default();
                copy_to_clipboard(copied.join(","));
            }
            return;
        }

        if query.get().is_empty() {
            match key.as_str() {
                "ArrowLeft" => {
                    if selected_items.is_empty() {
                        return;
                    }
                    ev.prevent_default();
                    let current = cursor_idx.get().unwrap_or(selected_items.len());
                    let next = current.saturating_sub(1);
                    if ev.shift_key() {
                        let anchor = selection_anchor.get().unwrap_or(next);
                        set_selection_anchor.set(Some(anchor));
                        set_selection_range.set(Some((anchor.min(next), anchor.max(next))));
                    } else {
                        set_selection_anchor.set(None);
                        set_selection_range.set(None);
                    }
                    set_cursor_idx.set(Some(next));
                    active.set(selected_items.get(next).cloned());
                    return;
                }
                "ArrowRight" => {
                    if selected_items.is_empty() {
                        return;
                    }
                    ev.prevent_default();
                    let current = cursor_idx.get().unwrap_or(0);
                    let next = (current + 1).min(selected_items.len());
                    if ev.shift_key() {
                        let anchor = selection_anchor.get().unwrap_or(current);
                        if next < selected_items.len() {
                            set_selection_anchor.set(Some(anchor));
                            set_selection_range.set(Some((anchor.min(next), anchor.max(next))));
                        } else {
                            set_selection_anchor.set(None);
                            set_selection_range.set(None);
                        }
                    } else {
                        set_selection_anchor.set(None);
                        set_selection_range.set(None);
                    }

                    if next == selected_items.len() {
                        set_cursor_idx.set(None);
                        active.set(None);
                    } else {
                        set_cursor_idx.set(Some(next));
                        active.set(selected_items.get(next).cloned());
                    }
                    return;
                }
                "Backspace" => {
                    ev.prevent_default();
                    if let Some((start, end)) = selection_range.get() {
                        selected.update(|s| {
                            s.drain(start..=end);
                        });
                        set_selection_anchor.set(None);
                        set_selection_range.set(None);
                        set_cursor_idx.set(None);
                        active.set(None);
                    } else if let Some(i) = cursor_idx.get() {
                        selected.update(|s| {
                            if i < s.len() {
                                s.remove(i);
                            }
                        });
                        let len = selected.get_untracked().len();
                        if len == 0 {
                            set_cursor_idx.set(None);
                            active.set(None);
                        } else {
                            let next = i.min(len - 1);
                            set_cursor_idx.set(Some(next));
                            active.set(selected.get_untracked().get(next).cloned());
                        }
                    } else {
                        selected.update(|s| {
                            if let Some(item) = s.pop() {
                                if pinned.get() == Some(item) {
                                    set_pinned.set(None);
                                }
                            }
                        });
                        active.set(None);
                    }
                    return;
                }
                _ => {}
            }
        }

        let suggs = suggestions.get();
        if suggs.is_empty() {
            if key == "Backspace" && query.get().is_empty() {
                selected.update(|s| {
                    if let Some(item) = s.pop() {
                        if pinned.get() == Some(item) {
                            set_pinned.set(None);
                            active.set(None);
                        }
                    }
                });
            }
            return;
        }

        match key.as_str() {
            "ArrowDown" => {
                ev.prevent_default();
                set_highlight_idx.update(|i| {
                    let mut next = *i + 1;
                    while next < suggs.len() {
                        let ok = is_selectable.is_none_or(|cb| cb.run(suggs[next].clone()));
                        if ok {
                            *i = next;
                            return;
                        }
                        next += 1;
                    }
                });
            }
            "ArrowUp" => {
                ev.prevent_default();
                set_highlight_idx.update(|i| {
                    if *i > 0 {
                        let mut prev = *i - 1;
                        loop {
                            let ok = is_selectable.is_none_or(|cb| cb.run(suggs[prev].clone()));
                            if ok {
                                *i = prev;
                                return;
                            }
                            if prev == 0 {
                                break;
                            }
                            prev -= 1;
                        }
                    }
                });
            }
            "Enter" => {
                ev.prevent_default();
                if let Some(item) = suggs.get(highlight_idx.get()) {
                    select(item.clone());
                }
            }
            _ => {}
        }
    };

    let on_pill_click = move |item: T| {
        if pinned.with(|p| p.as_ref() == Some(&item)) {
            set_pinned.set(None);
        } else {
            set_pinned.set(Some(item.clone()));
            active.set(Some(item));
        }
    };

    view! {
        <div
            class="search-container"
            on:click=move |_| {
                if let Some(input) = input_ref.get() {
                    input.focus().ok();
                }
            }
        >
            <div class="pills">
                <For
                    each=move || keyed_items(selected.get(), key)
                    key=|(stable_key, _, _)| *stable_key
                    children=move |(_, idx, item)| {
                        let item_for_active = item.clone();
                        let item_for_enter = item.clone();
                        let item_for_click = item.clone();
                        let render_pill = render_pill;
                        view! {
                            <div
                                class="pill"
                                class:active=move || {
                                    active.with(|a| a.as_ref() == Some(&item_for_active))
                                }
                                class:cursor=move || cursor_idx.get() == Some(idx)
                                class:selected=move || {
                                    selection_range
                                        .get()
                                        .is_some_and(|(start, end)| idx >= start && idx <= end)
                                }
                                on:mouseenter=move |_| active.set(Some(item_for_enter.clone()))
                                on:mouseleave=move |_| active.set(pinned.get())
                                on:click=move |ev| {
                                    ev.stop_propagation();
                                    set_cursor_idx.set(Some(idx));
                                    if ev.shift_key() {
                                        let anchor = selection_anchor.get().unwrap_or(idx);
                                        set_selection_anchor.set(Some(anchor));
                                        set_selection_range
                                            .set(Some((anchor.min(idx), anchor.max(idx))));
                                    } else {
                                        set_selection_anchor.set(Some(idx));
                                        set_selection_range.set(None);
                                    }
                                    on_pill_click(item_for_click.clone());
                                }
                            >
                                {render_pill(
                                    item.clone(),
                                    Callback::new(move |_| remove(item.clone())),
                                )}
                            </div>
                        }
                    }
                />
                <div class="input-wrapper">
                    <input
                        node_ref=input_ref
                        type="text"
                        placeholder=move || {
                            if selected.get().is_empty() {
                                placeholder.clone()
                            } else {
                                String::new()
                            }
                        }
                        prop:value=query
                        on:input=move |ev| {
                            let value = event_target_value(&ev);
                            parse_and_apply_csv(value);
                        }
                        on:focus=move |_| set_focused.set(true)
                        on:blur=move |_| {
                            set_timeout(move || set_focused.set(false), Duration::from_millis(150));
                        }
                        on:keydown=on_keydown
                    />
                </div>
            </div>

            <Show when=move || is_focused.get() && !suggestions.with(|s| s.is_empty())>
                <div class="dropdown">
                    <For
                        each=move || keyed_items(suggestions.get(), key)
                        key=|(stable_key, _, _)| *stable_key
                        children=move |(_, i, item)| {
                            let item_for_click = item.clone();
                            let item_for_update = item.clone();
                            let render_option = render_option;
                            let select = select;
                            let update_item = move |new_item: T| {
                                suggestions_override
                                    .update(|ov| {
                                        let mut current = ov
                                            .take()
                                            .unwrap_or_else(|| base_suggestions.get());
                                        if let Some(pos) = current
                                            .iter()
                                            .position(|x| *x == item_for_update)
                                        {
                                            current[pos] = new_item;
                                        }
                                        *ov = Some(current);
                                    });
                            };
                            let update_all = move |transform: Box<dyn Fn(T) -> T + Send + Sync>| {
                                suggestions_override
                                    .update(|ov| {
                                        let current = ov
                                            .take()
                                            .unwrap_or_else(|| base_suggestions.get());
                                        let updated: Vec<T> = current
                                            .into_iter()
                                            .map(&transform)
                                            .collect();
                                        *ov = Some(updated);
                                    });
                            };

                            view! {
                                <div
                                    class="option"
                                    class:selected=move || i == highlight_idx.get()
                                    on:click=move |_| select(item_for_click.clone())
                                    on:mouseenter=move |_| set_highlight_idx.set(i)
                                    on:mousedown=move |ev| ev.prevent_default()
                                >
                                    {render_option(
                                        item,
                                        i,
                                        suggestions,
                                        highlight_idx,
                                        Callback::new(select),
                                        Callback::new(update_item),
                                        Callback::new(update_all),
                                    )}
                                </div>
                            }
                        }
                    />
                </div>
            </Show>
        </div>
    }
}
