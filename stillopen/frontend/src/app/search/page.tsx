import SearchResults from "../../components/SearchResults";
import CitySearchResults from "../../components/CitySearchResults";

export default async function SearchPage({
    searchParams,
}: {
    searchParams: Promise<{ [key: string]: string | string[] | undefined }>
}) {
    const params = await searchParams;
    const query = typeof params.q === 'string' ? params.q : '';
    // Accept ?city= (new) or legacy ?l= (old) as the city/location filter
    const city = typeof params.city === 'string'
        ? params.city
        : typeof params.l === 'string'
        ? params.l
        : '';
    const page = typeof params.page === 'string'
        ? Math.max(1, parseInt(params.page, 10) || 1)
        : 1;
    const parentCategory = typeof params.parent_category === 'string' ? params.parent_category : undefined;

    if (city) {
        return (
            <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <CitySearchResults
                    query={query}
                    city={city}
                    initialPage={page}
                    initialParentCategory={parentCategory}
                />
            </div>
        );
    }

    return (
        <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            {(query || parentCategory) && (
                <div className="mb-8">
                    <h1 className="text-3xl font-black text-gray-900 dark:text-white tracking-tight">
                        {parentCategory && !query ? (
                            <>Browse <span className="text-emerald-500">{parentCategory}</span></>
                        ) : (
                            <>Results for &quot;<span className="text-emerald-500">{query}</span>&quot;</>
                        )}
                    </h1>
                </div>
            )}
            <SearchResults query={query} initialPage={page} initialParentCategory={parentCategory} />
        </div>
    );
}
